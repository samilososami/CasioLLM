/*
   Stateless TinyTalk v1 runtime for the Casio fx-CG50.

   The model is the official TheREZOR/TinyTalk GPT-Neo checkpoint converted
   with cardputer-ai's CRDP v3 Q4_0 exporter. We keep the 1.91 MB weights in
   storage and stream matrix rows through a small buffer. The CTK2 tokenizer
   is small enough to load once, which keeps prompt encoding and token decode
   responsive.

   It exposes its own tinytalk_* API so the dual-model add-in can link both
   runtimes while preserving the tested chat UI, streaming and cancellation.
*/
#include "tinytalk.h"

#include <gint/bfile.h>
#include <gint/gint.h>
#include <gint/rtc.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM 128
#define HIDDEN_DIM 512
#define LAYERS 8
#define HEADS 16
#define HEAD_SIZE 8
#define GROUPS (DIM / 32)
#define MAX_CONTEXT 64
#define MAX_POSITION 80
#define MAX_PROMPT_TOKENS 96
#define MAX_RESPONSE_TOKENS 32
#define VOCAB_SIZE 12062
#define TOKENIZER_MAX_BYTES 240000
#ifndef IO_BUFFER_BYTES
#define IO_BUFFER_BYTES 24000
#endif

#define TOK_MAGIC 0x324b5443U
#define PREFIX_TOKENS 2
#define PREFIX_HEADER_BYTES 48
#define PREFIX_MODEL_FINGERPRINT 0xc672d762U
#define PREFIX_PAYLOAD_BYTES \
    (LAYERS * PREFIX_TOKENS * ((DIM / 2) * 2 + GROUPS * 2 * 2))

static int const fixed_prefix_tokens[PREFIX_TOKENS] = { 7046, 25 };

typedef struct {
    uint32_t ln1_g, ln1_b, ln2_g, ln2_b;
    uint32_t lnf_g, lnf_b, bo, b_fc, b_proj, wpe;
    uint32_t wte, wq, wk, wv, wo, w_fc, w_proj, model_size;
    uint8_t *tokenizer;
    uint32_t tokenizer_size;
    uint32_t pieces_offset;
    int eos_id;
    int n_merges;
    int ready;
    char error[80];
} assets_t;

typedef struct {
    float x[DIM];
    float xb[DIM];
    float xb2[DIM];
    float q[DIM];
    float k[DIM];
    float v[DIM];
    float hb[HIDDEN_DIM];
    float att[HEADS * MAX_CONTEXT];
    int8_t xq[HIDDEN_DIM];
    float x_scales[HIDDEN_DIM / 32];
    uint8_t key_cache[LAYERS * MAX_CONTEXT * (DIM / 2)];
    uint8_t value_cache[LAYERS * MAX_CONTEXT * (DIM / 2)];
    uint16_t key_scales[LAYERS * MAX_CONTEXT * GROUPS];
    uint16_t value_scales[LAYERS * MAX_CONTEXT * GROUPS];
    int prompt[MAX_PROMPT_TOKENS];
    int prompt_len;
    int prompt_at;
    int position;
    int generated;
    int pending;
    int have_pending;
    int active;
    int forward_active;
    int forward_kind;
    int forward_token;
    int forward_position;
    int forward_layer;
    int forward_stage;
    int matrix_row;
    int choosing;
    int choose_at;
    int choose_best_id;
    float choose_best;
    int error_logged;
    uint32_t started_ticks;
    uint32_t first_token_ticks;
    uint32_t profile_switches;
    uint32_t profile_units;
    int first_token_recorded;
    char decoded[64];
} state_t;

enum { FORWARD_PROMPT = 1, FORWARD_EMIT = 2 };

enum {
    FWD_EMBED,
    FWD_LN1,
    FWD_Q,
    FWD_K,
    FWD_V,
    FWD_ATTENTION,
    FWD_O,
    FWD_ATT_RESIDUAL,
    FWD_LN2,
    FWD_FC,
    FWD_GELU,
    FWD_PROJ,
    FWD_FF_RESIDUAL,
    FWD_FINAL_NORM,
};

static assets_t assets;
static state_t state;
static uint8_t *io_buffer;
static char log_buffer[192];

static uint32_t elapsed_ticks(uint32_t start, uint32_t end)
{
    return end - start;
}

static void release_allocations(void)
{
    free(assets.tokenizer);
    free(io_buffer);
    assets.tokenizer = NULL;
    io_buffer = NULL;
}

static uint16_t le16(uint8_t const *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t le32(uint8_t const *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
        ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

#ifdef CASIOLLM_HOST_TOOLS
static void store_le32(uint8_t *p, uint32_t value)
{
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
    p[2] = (uint8_t)(value >> 16);
    p[3] = (uint8_t)(value >> 24);
}
#endif

static uint32_t prefix_token_hash(void)
{
    uint32_t hash = 2166136261U;
    for(int i = 0; i < PREFIX_TOKENS; i++) {
        uint32_t token = (uint32_t)fixed_prefix_tokens[i];
        for(int byte = 0; byte < 4; byte++) {
            hash ^= (token >> (byte * 8)) & 0xffU;
            hash *= 16777619U;
        }
    }
    return hash;
}

static float le_float(uint8_t const *p)
{
    uint32_t bits = le32(p);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float bf16_to_float(uint16_t value)
{
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t float_to_bf16(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return (uint16_t)(bits >> 16);
}

static void set_error(char const *text)
{
    strncpy(assets.error, text, sizeof(assets.error) - 1);
    assets.error[sizeof(assets.error) - 1] = '\0';
}

static int os_append_log(int ignored)
{
    static uint16_t const path[] = u"\\\\fls0\\CASIOLLM.LOG";
    char line[208];
    int fd;
    int size;
    int length;
    (void)ignored;

    fd = BFile_Open(path, BFile_ReadWrite);
    if(fd < 0) {
        size = 0;
        if(BFile_Create(path, BFile_File, &size) < 0) return 0;
        fd = BFile_Open(path, BFile_ReadWrite);
        if(fd < 0) return 0;
    }
    size = BFile_Size(fd);
    if(size < 0) { BFile_Close(fd); return 0; }
    if(size > 8192) {
        BFile_Close(fd);
        BFile_Remove(path);
        size = 0;
        if(BFile_Create(path, BFile_File, &size) < 0) return 0;
        fd = BFile_Open(path, BFile_ReadWrite);
        if(fd < 0) return 0;
    }
    length = (int)strlen(log_buffer);
    if(length > (int)sizeof(line) - 4) length = (int)sizeof(line) - 4;
    memcpy(line, log_buffer, (size_t)length);
    if((length + 2) & 1) line[length++] = ' ';
    line[length++] = '\r';
    line[length++] = '\n';
    if(BFile_Seek(fd, size) < 0 || BFile_Write(fd, line, length) != length) {
        BFile_Close(fd);
        return 0;
    }
    BFile_Close(fd);
    return 1;
}

static void log_event(char const *text)
{
    strncpy(log_buffer, text, sizeof(log_buffer) - 1);
    log_buffer[sizeof(log_buffer) - 1] = '\0';
    (void)gint_world_switch(GINT_CALL(os_append_log, 0));
}

static size_t row_stride(int columns)
{
    int blocks = columns / 32;
    int scale_bytes = blocks * 2;
    int padding = (-scale_bytes) & 15;
    return (size_t)(scale_bytes + padding + blocks * 16);
}

static uint32_t align16(uint32_t value)
{
    return (value + 15U) & ~15U;
}

static int read_exact(int fd, void *out, int bytes, uint32_t offset)
{
    return BFile_Read(fd, out, bytes, (int)offset) == bytes;
}

static int validate_tokenizer(void)
{
    uint8_t const *tok = assets.tokenizer;
    uint32_t at;
    if(assets.tokenizer_size < 532 || le32(tok) != TOK_MAGIC) return 0;
    if((int)le32(tok + 4) != VOCAB_SIZE) return 0;
    assets.eos_id = (int)le32(tok + 12);
    assets.n_merges = (int)le32(tok + 16);
    if(assets.eos_id < 0 || assets.eos_id >= VOCAB_SIZE || assets.n_merges < 0)
        return 0;
    at = 20U + 512U + (uint32_t)assets.n_merges * 6U;
    if(at > assets.tokenizer_size) return 0;
    assets.pieces_offset = at;
    for(int i = 0; i < VOCAB_SIZE; i++) {
        uint32_t length;
        if(at + 4U > assets.tokenizer_size) return 0;
        length = le32(tok + at);
        at += 4U;
        if(length > assets.tokenizer_size - at) return 0;
        at += length;
    }
    return at == assets.tokenizer_size;
}

static int os_load_assets(int ignored)
{
    static uint16_t const model_path[] = u"\\\\fls0\\TINYTLK.BIN";
    static uint16_t const tok_path[] = u"\\\\fls0\\TINYTLK.TOK";
    uint8_t header[64];
    uint32_t offset;
    uint32_t layer_dim = LAYERS * DIM;
    int fd;
    int tok_size;
    (void)ignored;

    fd = BFile_Open(model_path, BFile_ReadOnly);
    if(fd < 0) { set_error("Missing TINYTLK.BIN"); return 0; }
    assets.model_size = (uint32_t)BFile_Size(fd);
    if(!read_exact(fd, header, sizeof(header), 0)) {
        BFile_Close(fd); set_error("Cannot read TINYTLK.BIN"); return 0;
    }
    BFile_Close(fd);
    if(memcmp(header, "CRDP", 4) != 0 || le32(header + 4) != 3 ||
       (int)le32(header + 8) != DIM || (int)le32(header + 12) != HIDDEN_DIM ||
       (int)le32(header + 16) != LAYERS || (int)le32(header + 20) != HEADS ||
       (int)le32(header + 28) != VOCAB_SIZE ||
       (int)le32(header + 32) != MAX_POSITION ||
       header[36] != 1 || header[37] != 4 || header[38] != 2) {
        set_error("Invalid TinyTalk model"); return 0;
    }

    offset = 64;
    assets.ln1_g = offset; offset += layer_dim * 4U;
    assets.ln1_b = offset; offset += layer_dim * 4U;
    assets.ln2_g = offset; offset += layer_dim * 4U;
    assets.ln2_b = offset; offset += layer_dim * 4U;
    assets.lnf_g = offset; offset += DIM * 4U;
    assets.lnf_b = offset; offset += DIM * 4U;
    assets.bo = offset; offset += layer_dim * 4U;
    assets.b_fc = offset; offset += LAYERS * HIDDEN_DIM * 4U;
    assets.b_proj = offset; offset += layer_dim * 4U;
    assets.wpe = offset; offset += MAX_POSITION * DIM * 4U;
    offset = align16(offset);
    assets.wte = offset; offset += VOCAB_SIZE * (uint32_t)row_stride(DIM);
    assets.wq = offset; offset += LAYERS * DIM * (uint32_t)row_stride(DIM);
    assets.wk = offset; offset += LAYERS * DIM * (uint32_t)row_stride(DIM);
    assets.wv = offset; offset += LAYERS * DIM * (uint32_t)row_stride(DIM);
    assets.wo = offset; offset += LAYERS * DIM * (uint32_t)row_stride(DIM);
    assets.w_fc = offset; offset += LAYERS * HIDDEN_DIM * (uint32_t)row_stride(DIM);
    assets.w_proj = offset;
    offset += LAYERS * DIM * (uint32_t)row_stride(HIDDEN_DIM);
    if(offset != assets.model_size) {
        set_error("TinyTalk model size mismatch"); return 0;
    }

    fd = BFile_Open(tok_path, BFile_ReadOnly);
    if(fd < 0) { set_error("Missing TINYTLK.TOK"); return 0; }
    tok_size = BFile_Size(fd);
    if(tok_size < 532 || tok_size > TOKENIZER_MAX_BYTES) {
        BFile_Close(fd); set_error("Invalid TINYTLK.TOK size"); return 0;
    }
    assets.tokenizer = malloc((size_t)tok_size);
    io_buffer = malloc(IO_BUFFER_BYTES);
    if(!assets.tokenizer || !io_buffer) {
        BFile_Close(fd);
        release_allocations();
        set_error("Not enough RAM for TinyTalk");
        return 0;
    }
    if(BFile_Read(fd, assets.tokenizer, tok_size, 0) != tok_size) {
        BFile_Close(fd);
        release_allocations();
        set_error("Cannot read TINYTLK.TOK");
        return 0;
    }
    BFile_Close(fd);
    assets.tokenizer_size = (uint32_t)tok_size;
    if(!validate_tokenizer()) {
        release_allocations();
        set_error("Invalid TINYTLK.TOK");
        return 0;
    }
    assets.ready = 1;
    set_error("");
    return 1;
}

bool tinytalk_prepare(void)
{
    if(assets.ready) return true;
    return gint_world_switch(GINT_CALL(os_load_assets, 0)) == 1;
}

static int os_load_prefix(int ignored)
{
    uint8_t header[PREFIX_HEADER_BYTES];
    uint8_t scale_raw[PREFIX_TOKENS * GROUPS * 2];
    int fd;
    int offset = PREFIX_HEADER_BYTES;
    int kv_bytes = PREFIX_TOKENS * (DIM / 2);
    int scale_bytes = PREFIX_TOKENS * GROUPS * 2;
    (void)ignored;

    fd = BFile_Open(u"\\\\fls0\\TINYTLK.PFX", BFile_ReadOnly);
    if(fd < 0) return 0;
    if(BFile_Size(fd) != PREFIX_HEADER_BYTES + PREFIX_PAYLOAD_BYTES ||
       BFile_Read(fd, header, sizeof(header), 0) != (int)sizeof(header) ||
       memcmp(header, "TTPFX001", 8) != 0 ||
       le32(header + 8) != PREFIX_MODEL_FINGERPRINT ||
       le32(header + 12) != assets.model_size ||
       le32(header + 16) != PREFIX_TOKENS ||
       le32(header + 20) != DIM || le32(header + 24) != LAYERS ||
       le32(header + 28) != GROUPS ||
       le32(header + 32) != PREFIX_PAYLOAD_BYTES ||
       le32(header + 36) != prefix_token_hash() ||
       le32(header + 40) != MAX_CONTEXT || le32(header + 44) != 1) {
        BFile_Close(fd);
        return 0;
    }
    for(int layer = 0; layer < LAYERS; layer++) {
        if(BFile_Read(fd, state.key_cache +
                (size_t)layer * MAX_CONTEXT * (DIM / 2), kv_bytes, offset)
                != kv_bytes) goto fail;
        offset += kv_bytes;
        if(BFile_Read(fd, state.value_cache +
                (size_t)layer * MAX_CONTEXT * (DIM / 2), kv_bytes, offset)
                != kv_bytes) goto fail;
        offset += kv_bytes;
        if(BFile_Read(fd, scale_raw, scale_bytes, offset) != scale_bytes)
            goto fail;
        for(int i = 0; i < PREFIX_TOKENS * GROUPS; i++)
            state.key_scales[(size_t)layer * MAX_CONTEXT * GROUPS + i] =
                le16(scale_raw + i * 2);
        offset += scale_bytes;
        if(BFile_Read(fd, scale_raw, scale_bytes, offset) != scale_bytes)
            goto fail;
        for(int i = 0; i < PREFIX_TOKENS * GROUPS; i++)
            state.value_scales[(size_t)layer * MAX_CONTEXT * GROUPS + i] =
                le16(scale_raw + i * 2);
        offset += scale_bytes;
    }
    BFile_Close(fd);
    return 1;

fail:
    BFile_Close(fd);
    return 0;
}

static int find_merge(int a, int b)
{
    uint8_t const *base = assets.tokenizer + 20 + 512;
    int low = 0;
    int high = assets.n_merges - 1;
    while(low <= high) {
        int middle = (low + high) >> 1;
        uint8_t const *entry = base + middle * 6;
        int ma = (int)le16(entry);
        int mb = (int)le16(entry + 2);
        if(ma < a || (ma == a && mb < b)) low = middle + 1;
        else if(ma > a || (ma == a && mb > b)) high = middle - 1;
        else return (int)le16(entry + 4);
    }
    return -1;
}

static int encode_text(char const *text, int *tokens, int capacity)
{
    uint8_t const *byte_ids = assets.tokenizer + 20;
    int count = 0;
    for(unsigned char const *p = (unsigned char const *)text; *p; p++) {
        int id;
        if(count >= capacity) return -1;
        id = (int)le16(byte_ids + (int)(*p) * 2);
        if(id < 0 || id >= VOCAB_SIZE) return -1;
        tokens[count++] = id;
    }
    while(count > 1) {
        int best = -1;
        int best_at = -1;
        for(int i = 0; i + 1 < count; i++) {
            int merged = find_merge(tokens[i], tokens[i + 1]);
            if(merged >= 0 && (best < 0 || merged < best)) {
                best = merged;
                best_at = i;
            }
        }
        if(best_at < 0) break;
        tokens[best_at] = best;
        memmove(tokens + best_at + 1, tokens + best_at + 2,
            (size_t)(count - best_at - 2) * sizeof(tokens[0]));
        count--;
    }
    return count;
}

static void decode_token_to_state(int token)
{
    uint8_t const *tok = assets.tokenizer;
    uint32_t at = assets.pieces_offset;
    uint32_t length = 0;
    int out = 0;
    state.decoded[0] = '\0';
    if(token < 0 || token >= VOCAB_SIZE || token == assets.eos_id) return;
    for(int i = 0; i <= token; i++) {
        length = le32(tok + at);
        at += 4;
        if(i == token) break;
        at += length;
    }
    for(uint32_t i = 0; i < length && out + 1 < (int)sizeof(state.decoded); i++) {
        unsigned char c = tok[at + i];
        if(c == '\n' || c == '\r' || c == '\t') c = ' ';
        if(c >= 32 && c < 127) state.decoded[out++] = (char)c;
    }
    state.decoded[out] = '\0';
}

bool tinytalk_start(char const *user_prompt)
{
    char formatted[128];
    char line[176];
    int length;
    int cached = 0;
    uint32_t request_started = rtc_ticks();
    if(!tinytalk_prepare()) return false;
    memset(&state, 0, sizeof(state));
#ifdef CASIOLLM_TINYTALK_IDENTITY_PROMPT
    snprintf(formatted, sizeof(formatted),
        "You are CasioLLM.\nUser: %.80s\nBot:", user_prompt);
#else
    snprintf(formatted, sizeof(formatted), "User: %.80s\nBot:", user_prompt);
#endif
    length = encode_text(formatted, state.prompt, MAX_PROMPT_TOKENS);
    if(length <= 0) { set_error("TinyTalk prompt encoding failed"); return false; }
    if(length >= PREFIX_TOKENS &&
       state.prompt[0] == fixed_prefix_tokens[0] &&
       state.prompt[1] == fixed_prefix_tokens[1] &&
       gint_world_switch(GINT_CALL(os_load_prefix, 0)) == 1) {
        memmove(state.prompt, state.prompt + PREFIX_TOKENS,
            (size_t)(length - PREFIX_TOKENS) * sizeof(state.prompt[0]));
        length -= PREFIX_TOKENS;
        state.position = PREFIX_TOKENS;
        cached = PREFIX_TOKENS;
    }
    state.prompt_len = length;
    state.active = 1;
    state.started_ticks = request_started;
    set_error("");
    snprintf(line, sizeof(line),
        "START backend=TinyTalk-v1 prompt_tokens=%d cached=%d user=%.80s",
        state.prompt_len, cached, user_prompt);
    log_event(line);
    return true;
}

static void quantize_q8(float const *input, int count)
{
    int blocks = count / 32;
    for(int block = 0; block < blocks; block++) {
        float maximum = 0.0f;
        float scale;
        float inverse;
        for(int i = 0; i < 32; i++) {
            float absolute = fabsf(input[block * 32 + i]);
            if(absolute > maximum) maximum = absolute;
        }
        scale = maximum / 127.0f;
        inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
        state.x_scales[block] = scale;
        for(int i = 0; i < 32; i++) {
            float scaled = input[block * 32 + i] * inverse;
            int value = (int)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
            if(value > 127) value = 127;
            if(value < -127) value = -127;
            state.xq[block * 32 + i] = (int8_t)value;
        }
    }
    state.matrix_row = 0;
}

static float dot_q4_q8(uint8_t const *row, int columns)
{
    int blocks = columns / 32;
    int scale_bytes = blocks * 2;
    int nibble_offset = scale_bytes + ((-scale_bytes) & 15);
    uint8_t const *nibbles = row + nibble_offset;
    float total = 0.0f;
    for(int block = 0; block < blocks; block++) {
        int32_t sum = 0;
        int base = block * 32;
        for(int i = 0; i < 16; i++) {
            uint8_t packed = nibbles[block * 16 + i];
            sum += ((int)(packed & 15) - 8) * (int)state.xq[base + i];
            sum += ((int)(packed >> 4) - 8) * (int)state.xq[base + i + 16];
        }
        total += (float)sum * bf16_to_float(le16(row + block * 2)) *
            state.x_scales[block];
    }
    return total;
}

static void dequant_q4_row(float *out, uint8_t const *row, int columns)
{
    int blocks = columns / 32;
    int scale_bytes = blocks * 2;
    int nibble_offset = scale_bytes + ((-scale_bytes) & 15);
    uint8_t const *nibbles = row + nibble_offset;
    for(int block = 0; block < blocks; block++) {
        float scale = bf16_to_float(le16(row + block * 2));
        int base = block * 32;
        for(int i = 0; i < 16; i++) {
            uint8_t packed = nibbles[block * 16 + i];
            out[base + i] = scale * (float)((int)(packed & 15) - 8);
            out[base + i + 16] = scale * (float)((int)(packed >> 4) - 8);
        }
    }
}

static int matvec_microstep(int fd, uint32_t offset, int rows, int columns,
    float *output)
{
    int remaining = rows - state.matrix_row;
    int stride = (int)row_stride(columns);
    int take = IO_BUFFER_BYTES / stride;
    int bytes;
    if(take > remaining) take = remaining;
    if(take <= 0) return -1;
    bytes = take * stride;
    if(!read_exact(fd, io_buffer, bytes,
        offset + (uint32_t)state.matrix_row * (uint32_t)stride)) return -1;
    for(int row = 0; row < take; row++)
        output[state.matrix_row + row] = dot_q4_q8(io_buffer + row * stride, columns);
    state.matrix_row += take;
    if(state.matrix_row < rows) return 0;
    state.matrix_row = 0;
    return 1;
}

static int layernorm_from_file(int fd, float const *input, float *output,
    uint32_t gamma_offset, uint32_t beta_offset, int count)
{
    float mean = 0.0f;
    float variance = 0.0f;
    float inverse;
    int bytes = count * 4;
    if(bytes * 2 > IO_BUFFER_BYTES ||
       !read_exact(fd, io_buffer, bytes, gamma_offset) ||
       !read_exact(fd, io_buffer + bytes, bytes, beta_offset)) return 0;
    for(int i = 0; i < count; i++) mean += input[i];
    mean /= (float)count;
    for(int i = 0; i < count; i++) {
        float difference = input[i] - mean;
        variance += difference * difference;
    }
    inverse = 1.0f / sqrtf(variance / (float)count + 1e-5f);
    for(int i = 0; i < count; i++) {
        float gamma = le_float(io_buffer + i * 4);
        float beta = le_float(io_buffer + bytes + i * 4);
        output[i] = gamma * ((input[i] - mean) * inverse) + beta;
    }
    return 1;
}

static int add_bias(int fd, float *target, uint32_t offset, int count,
    float const *residual)
{
    int bytes = count * 4;
    if(bytes > IO_BUFFER_BYTES || !read_exact(fd, io_buffer, bytes, offset)) return 0;
    for(int i = 0; i < count; i++)
        target[i] += le_float(io_buffer + i * 4) + (residual ? residual[i] : 0.0f);
    return 1;
}

static void cache_current_kv(void)
{
    int layer = state.forward_layer;
    int pos = state.forward_position;
    size_t row_at = ((size_t)layer * MAX_CONTEXT + pos) * (DIM / 2);
    size_t scale_at = ((size_t)layer * MAX_CONTEXT + pos) * GROUPS;
    uint8_t *key_row = state.key_cache + row_at;
    uint8_t *value_row = state.value_cache + row_at;
    for(int group = 0; group < GROUPS; group++) {
        float key_max = 0.0f, key_signed = 0.0f;
        float value_max = 0.0f, value_signed = 0.0f;
        float key_scale, value_scale, key_inverse, value_inverse;
        for(int i = 0; i < 32; i++) {
            float ka = fabsf(state.k[group * 32 + i]);
            float va = fabsf(state.v[group * 32 + i]);
            if(ka > key_max) { key_max = ka; key_signed = state.k[group * 32 + i]; }
            if(va > value_max) { value_max = va; value_signed = state.v[group * 32 + i]; }
        }
        state.key_scales[scale_at + group] = float_to_bf16(key_signed / -8.0f);
        state.value_scales[scale_at + group] = float_to_bf16(value_signed / -8.0f);
        key_scale = bf16_to_float(state.key_scales[scale_at + group]);
        value_scale = bf16_to_float(state.value_scales[scale_at + group]);
        key_inverse = key_scale != 0.0f ? 1.0f / key_scale : 0.0f;
        value_inverse = value_scale != 0.0f ? 1.0f / value_scale : 0.0f;
        for(int i = 0; i < 32; i += 2) {
            int k0 = (int)(state.k[group * 32 + i] * key_inverse +
                (state.k[group * 32 + i] * key_inverse >= 0.0f ? 0.5f : -0.5f));
            int k1 = (int)(state.k[group * 32 + i + 1] * key_inverse +
                (state.k[group * 32 + i + 1] * key_inverse >= 0.0f ? 0.5f : -0.5f));
            int v0 = (int)(state.v[group * 32 + i] * value_inverse +
                (state.v[group * 32 + i] * value_inverse >= 0.0f ? 0.5f : -0.5f));
            int v1 = (int)(state.v[group * 32 + i + 1] * value_inverse +
                (state.v[group * 32 + i + 1] * value_inverse >= 0.0f ? 0.5f : -0.5f));
            if(k0 > 7) k0 = 7;
            if(k0 < -8) k0 = -8;
            if(k1 > 7) k1 = 7;
            if(k1 < -8) k1 = -8;
            if(v0 > 7) v0 = 7;
            if(v0 < -8) v0 = -8;
            if(v1 > 7) v1 = 7;
            if(v1 < -8) v1 = -8;
            key_row[group * 16 + i / 2] = (uint8_t)((k0 + 8) | ((k1 + 8) << 4));
            value_row[group * 16 + i / 2] = (uint8_t)((v0 + 8) | ((v1 + 8) << 4));
        }
    }
}

static void softmax(float *values, int count)
{
    float maximum = values[0];
    float sum = 0.0f;
    for(int i = 1; i < count; i++) if(values[i] > maximum) maximum = values[i];
    for(int i = 0; i < count; i++) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    for(int i = 0; i < count; i++) values[i] /= sum;
}

static void compute_attention(void)
{
    int layer = state.forward_layer;
    int pos = state.forward_position;
    memset(state.xb, 0, sizeof(state.xb));
    for(int head = 0; head < HEADS; head++) {
        float *scores = state.att + head * MAX_CONTEXT;
        int group = (head * HEAD_SIZE) / 32;
        int nibble = (head * HEAD_SIZE) / 2;
        for(int time = 0; time <= pos; time++) {
            size_t row_at = ((size_t)layer * MAX_CONTEXT + time) * (DIM / 2);
            size_t scale_at = ((size_t)layer * MAX_CONTEXT + time) * GROUPS;
            uint8_t const *key = state.key_cache + row_at + nibble;
            float score = 0.0f;
            for(int i = 0; i < HEAD_SIZE; i += 2) {
                uint8_t packed = key[i / 2];
                score += state.q[head * HEAD_SIZE + i] * (float)((int)(packed & 15) - 8);
                score += state.q[head * HEAD_SIZE + i + 1] * (float)((int)(packed >> 4) - 8);
            }
            scores[time] = score * bf16_to_float(state.key_scales[scale_at + group]);
        }
        softmax(scores, pos + 1);
        for(int time = 0; time <= pos; time++) {
            size_t row_at = ((size_t)layer * MAX_CONTEXT + time) * (DIM / 2);
            size_t scale_at = ((size_t)layer * MAX_CONTEXT + time) * GROUPS;
            uint8_t const *value = state.value_cache + row_at + nibble;
            float weight = scores[time] *
                bf16_to_float(state.value_scales[scale_at + group]);
            for(int i = 0; i < HEAD_SIZE; i += 2) {
                uint8_t packed = value[i / 2];
                state.xb[head * HEAD_SIZE + i] +=
                    weight * (float)((int)(packed & 15) - 8);
                state.xb[head * HEAD_SIZE + i + 1] +=
                    weight * (float)((int)(packed >> 4) - 8);
            }
        }
    }
}

static int forward_begin(int token, int kind)
{
    if(token < 0 || token >= VOCAB_SIZE || state.position >= MAX_CONTEXT) return 0;
    state.forward_active = 1;
    state.forward_kind = kind;
    state.forward_token = token;
    state.forward_position = state.position;
    state.forward_layer = 0;
    state.forward_stage = FWD_EMBED;
    state.matrix_row = 0;
    return 1;
}

static int forward_microstep(int fd)
{
    int layer = state.forward_layer;
    int result;
    uint32_t layer_dim_offset = (uint32_t)layer * DIM * 4U;
    uint32_t matrix128 = (uint32_t)layer * DIM * (uint32_t)row_stride(DIM);

    switch(state.forward_stage) {
        case FWD_EMBED:
            if(!read_exact(fd, io_buffer, (int)row_stride(DIM),
                assets.wte + (uint32_t)state.forward_token * (uint32_t)row_stride(DIM)))
                return -1;
            dequant_q4_row(state.x, io_buffer, DIM);
            if(!read_exact(fd, io_buffer, DIM * 4,
                assets.wpe + (uint32_t)state.forward_position * DIM * 4U)) return -1;
            for(int i = 0; i < DIM; i++) state.x[i] += le_float(io_buffer + i * 4);
            state.forward_stage = FWD_LN1;
            return 0;

        case FWD_LN1:
            if(!layernorm_from_file(fd, state.x, state.xb,
                assets.ln1_g + layer_dim_offset, assets.ln1_b + layer_dim_offset, DIM))
                return -1;
            quantize_q8(state.xb, DIM);
            state.forward_stage = FWD_Q;
            return 0;

        case FWD_Q:
            result = matvec_microstep(fd, assets.wq + matrix128, DIM, DIM, state.q);
            if(result <= 0) return result;
            quantize_q8(state.xb, DIM);
            state.forward_stage = FWD_K;
            return 0;

        case FWD_K:
            result = matvec_microstep(fd, assets.wk + matrix128, DIM, DIM, state.k);
            if(result <= 0) return result;
            quantize_q8(state.xb, DIM);
            state.forward_stage = FWD_V;
            return 0;

        case FWD_V:
            result = matvec_microstep(fd, assets.wv + matrix128, DIM, DIM, state.v);
            if(result <= 0) return result;
            cache_current_kv();
            state.forward_stage = FWD_ATTENTION;
            return 0;

        case FWD_ATTENTION:
            compute_attention();
            quantize_q8(state.xb, DIM);
            state.forward_stage = FWD_O;
            return 0;

        case FWD_O:
            result = matvec_microstep(fd, assets.wo + matrix128, DIM, DIM, state.xb2);
            if(result <= 0) return result;
            state.forward_stage = FWD_ATT_RESIDUAL;
            return 0;

        case FWD_ATT_RESIDUAL:
            if(!add_bias(fd, state.xb2, assets.bo + layer_dim_offset, DIM, state.x))
                return -1;
            memcpy(state.x, state.xb2, sizeof(state.x));
            state.forward_stage = FWD_LN2;
            return 0;

        case FWD_LN2:
            if(!layernorm_from_file(fd, state.x, state.xb,
                assets.ln2_g + layer_dim_offset, assets.ln2_b + layer_dim_offset, DIM))
                return -1;
            quantize_q8(state.xb, DIM);
            state.forward_stage = FWD_FC;
            return 0;

        case FWD_FC:
            result = matvec_microstep(fd,
                assets.w_fc + (uint32_t)layer * HIDDEN_DIM * (uint32_t)row_stride(DIM),
                HIDDEN_DIM, DIM, state.hb);
            if(result <= 0) return result;
            state.forward_stage = FWD_GELU;
            return 0;

        case FWD_GELU:
            if(!read_exact(fd, io_buffer, HIDDEN_DIM * 4,
                assets.b_fc + (uint32_t)layer * HIDDEN_DIM * 4U)) return -1;
            for(int i = 0; i < HIDDEN_DIM; i++) {
                float value = state.hb[i] + le_float(io_buffer + i * 4);
                state.hb[i] = 0.5f * value * (1.0f +
                    tanhf(0.7978845608f * (value + 0.044715f * value * value * value)));
            }
            quantize_q8(state.hb, HIDDEN_DIM);
            state.forward_stage = FWD_PROJ;
            return 0;

        case FWD_PROJ:
            result = matvec_microstep(fd,
                assets.w_proj + (uint32_t)layer * DIM * (uint32_t)row_stride(HIDDEN_DIM),
                DIM, HIDDEN_DIM, state.xb);
            if(result <= 0) return result;
            state.forward_stage = FWD_FF_RESIDUAL;
            return 0;

        case FWD_FF_RESIDUAL:
            if(!add_bias(fd, state.xb, assets.b_proj + layer_dim_offset, DIM, state.x))
                return -1;
            memcpy(state.x, state.xb, sizeof(state.x));
            state.forward_layer++;
            if(state.forward_layer < LAYERS) state.forward_stage = FWD_LN1;
            else state.forward_stage = FWD_FINAL_NORM;
            return 0;

        case FWD_FINAL_NORM:
            if(!layernorm_from_file(fd, state.x, state.xb,
                assets.lnf_g, assets.lnf_b, DIM)) return -1;
            memcpy(state.x, state.xb, sizeof(state.x));
            state.position++;
            state.forward_active = 0;
            return 1;

        default:
            return -1;
    }
}

static void choose_token_begin(void)
{
    quantize_q8(state.x, DIM);
    state.choosing = 1;
    state.choose_at = 0;
    state.choose_best_id = 0;
    state.choose_best = -1e30f;
}

static int choose_token_microstep(int fd)
{
    int stride = (int)row_stride(DIM);
    int remaining = VOCAB_SIZE - state.choose_at;
    int take = IO_BUFFER_BYTES / stride;
    int bytes;
    if(take > remaining) take = remaining;
    if(take <= 0) return -1;
    bytes = take * stride;
    if(!read_exact(fd, io_buffer, bytes,
        assets.wte + (uint32_t)state.choose_at * (uint32_t)stride)) return -1;
    for(int i = 0; i < take; i++) {
        float score = dot_q4_q8(io_buffer + i * stride, DIM);
        if(score > state.choose_best) {
            state.choose_best = score;
            state.choose_best_id = state.choose_at + i;
        }
    }
    state.choose_at += take;
    if(state.choose_at < VOCAB_SIZE) return 0;
    state.choosing = 0;
    return 1;
}

static int os_step_unit(int fd)
{
    int result;

    if(state.forward_active) {
        result = forward_microstep(fd);
        if(result < 0) { set_error("TinyTalk read or compute error"); return -2; }
        if(result == 0) return 0;
        if(state.prompt_at < state.prompt_len) return 0;
        choose_token_begin();
        return 0;
    }

    if(state.choosing) {
        result = choose_token_microstep(fd);
        if(result < 0) { set_error("TinyTalk vocabulary read error"); return -2; }
        if(result == 0) return 0;
        state.pending = state.choose_best_id;
        if(state.pending == assets.eos_id) return -1;
        state.have_pending = 1;
        state.generated++;
        decode_token_to_state(state.pending);
        return state.pending + 1;
    }

    if(state.prompt_at < state.prompt_len) {
        int token = state.prompt[state.prompt_at++];
        int ok = forward_begin(token, FORWARD_PROMPT);
        if(!ok) { set_error("TinyTalk context error"); return -2; }
        return 0;
    }
    if(state.have_pending) {
        int token = state.pending;
        state.have_pending = 0;
        if(state.generated >= MAX_RESPONSE_TOKENS) {
            return -1;
        }
        if(!forward_begin(token, FORWARD_EMIT)) {
            set_error("TinyTalk context error"); return -2;
        }
        return 0;
    }
    return -1;
}

/* Extreme cooperative mode: keep the model open and execute a bounded batch
   of legacy units per OS visit. The build selects the batch size according to
   the accepted F6 latency budget. */
static int os_step_extreme(int ignored)
{
    static uint16_t const path[] = u"\\\\fls0\\TINYTLK.BIN";
    int fd;
    int result = 0;
    (void)ignored;
    fd = BFile_Open(path, BFile_ReadOnly);
    if(fd < 0) { set_error("Missing TINYTLK.BIN"); return -2; }
    for(int i = 0; i < CASIOLLM_TINYTALK_COALESCE_STEPS && result == 0; i++) {
        state.profile_units++;
        result = os_step_unit(fd);
    }
    BFile_Close(fd);
    return result;
}

int tinytalk_step(int *token_id)
{
    int result;
    char line[176];
    if(!state.active) return -1;
    state.profile_switches++;
    result = gint_world_switch(GINT_CALL(os_step_extreme, 0));
    if(result > 0) {
        *token_id = result - 1;
        if(!state.first_token_recorded) {
            state.first_token_ticks = elapsed_ticks(state.started_ticks, rtc_ticks());
            state.first_token_recorded = 1;
        }
        snprintf(line, sizeof(line),
            "TOKEN backend=TinyTalk-v1 n=%d id=%d pos=%d text=%.48s",
            state.generated, *token_id, state.position, state.decoded);
        log_event(line);
        return 1;
    }
    if(result < 0) {
        state.active = 0;
        if(result == -2 && !state.error_logged) {
            snprintf(line, sizeof(line),
                "ERROR backend=TinyTalk-v1 %s prompt=%d/%d pos=%d layer=%d stage=%d",
                assets.error, state.prompt_at, state.prompt_len, state.position,
                state.forward_layer, state.forward_stage);
            state.error_logged = 1;
            log_event(line);
        }
        else if(result == -1) {
            uint32_t total = elapsed_ticks(state.started_ticks, rtc_ticks());
            snprintf(line, sizeof(line),
                "END backend=TinyTalk-v1 gen=%d pos=%d first_ms=%lu total_ms=%lu sw=%lu u=%lu",
                state.generated, state.position,
                (unsigned long)(state.first_token_ticks * 1000U / 128U),
                (unsigned long)(total * 1000U / 128U),
                (unsigned long)state.profile_switches,
                (unsigned long)state.profile_units);
            log_event(line);
        }
        return -1;
    }
    return 0;
}

void tinytalk_cancel(void)
{
    char line[176];
    uint32_t total = elapsed_ticks(state.started_ticks, rtc_ticks());
    state.active = 0;
    state.have_pending = 0;
    state.forward_active = 0;
    state.choosing = 0;
    snprintf(line, sizeof(line),
        "CANCEL backend=TinyTalk-v1 p=%d/%d pos=%d l=%d s=%d gen=%d first_ms=%lu total_ms=%lu sw=%lu u=%lu",
        state.prompt_at, state.prompt_len, state.position,
        state.forward_layer, state.forward_stage, state.generated,
        (unsigned long)(state.first_token_ticks * 1000U / 128U),
        (unsigned long)(total * 1000U / 128U),
        (unsigned long)state.profile_switches,
        (unsigned long)state.profile_units);
    log_event(line);
}

void tinytalk_decode_token(int token_id, char *out, size_t out_size)
{
    (void)token_id;
    if(!out_size) return;
    strncpy(out, state.decoded, out_size - 1);
    out[out_size - 1] = '\0';
}

char const *tinytalk_error(void)
{
    return assets.error;
}

void tinytalk_shutdown(void)
{
    if(state.active) tinytalk_cancel();
    release_allocations();
    memset(&state, 0, sizeof(state));
    memset(&assets, 0, sizeof(assets));
}

#ifdef CASIOLLM_HOST_TOOLS
int tinytalk_debug_encode(char const *text, int *tokens, int capacity)
{
    if(!tinytalk_prepare()) return -1;
    return encode_text(text, tokens, capacity);
}

void tinytalk_debug_decode(int token_id, char *out, size_t out_size)
{
    decode_token_to_state(token_id);
    if(!out_size) return;
    strncpy(out, state.decoded, out_size - 1);
    out[out_size - 1] = '\0';
}

static int os_debug_compute_prefix(int ignored)
{
    int fd;
    (void)ignored;
    fd = BFile_Open(u"\\\\fls0\\TINYTLK.BIN", BFile_ReadOnly);
    if(fd < 0) { set_error("Missing TINYTLK.BIN"); return 0; }
    while(state.prompt_at < state.prompt_len || state.forward_active) {
        if(!state.forward_active) {
            int token = state.prompt[state.prompt_at++];
            if(!forward_begin(token, FORWARD_PROMPT)) {
                BFile_Close(fd);
                set_error("Prefix context error");
                return 0;
            }
        }
        else if(forward_microstep(fd) < 0) {
            BFile_Close(fd);
            set_error("Prefix compute error");
            return 0;
        }
    }
    BFile_Close(fd);
    return state.position == PREFIX_TOKENS;
}

static int os_debug_write_prefix(int ignored)
{
    uint8_t header[PREFIX_HEADER_BYTES] = { 0 };
    uint8_t scale_raw[PREFIX_TOKENS * GROUPS * 2];
    int empty_size = 0;
    int fd;
    int kv_bytes = PREFIX_TOKENS * (DIM / 2);
    int scale_bytes = PREFIX_TOKENS * GROUPS * 2;
    (void)ignored;

    memcpy(header, "TTPFX001", 8);
    store_le32(header + 8, PREFIX_MODEL_FINGERPRINT);
    store_le32(header + 12, assets.model_size);
    store_le32(header + 16, PREFIX_TOKENS);
    store_le32(header + 20, DIM);
    store_le32(header + 24, LAYERS);
    store_le32(header + 28, GROUPS);
    store_le32(header + 32, PREFIX_PAYLOAD_BYTES);
    store_le32(header + 36, prefix_token_hash());
    store_le32(header + 40, MAX_CONTEXT);
    store_le32(header + 44, 1);

    (void)BFile_Remove(u"\\\\fls0\\TINYTLK.PFX");
    if(BFile_Create(u"\\\\fls0\\TINYTLK.PFX", BFile_File, &empty_size) < 0)
        return 0;
    fd = BFile_Open(u"\\\\fls0\\TINYTLK.PFX", BFile_ReadWrite);
    if(fd < 0) return 0;
    if(BFile_Write(fd, header, sizeof(header)) != (int)sizeof(header)) goto fail;
    for(int layer = 0; layer < LAYERS; layer++) {
        if(BFile_Write(fd, state.key_cache +
                (size_t)layer * MAX_CONTEXT * (DIM / 2), kv_bytes) != kv_bytes)
            goto fail;
        if(BFile_Write(fd, state.value_cache +
                (size_t)layer * MAX_CONTEXT * (DIM / 2), kv_bytes) != kv_bytes)
            goto fail;
        for(int i = 0; i < PREFIX_TOKENS * GROUPS; i++) {
            uint16_t value =
                state.key_scales[(size_t)layer * MAX_CONTEXT * GROUPS + i];
            scale_raw[i * 2] = (uint8_t)value;
            scale_raw[i * 2 + 1] = (uint8_t)(value >> 8);
        }
        if(BFile_Write(fd, scale_raw, scale_bytes) != scale_bytes) goto fail;
        for(int i = 0; i < PREFIX_TOKENS * GROUPS; i++) {
            uint16_t value =
                state.value_scales[(size_t)layer * MAX_CONTEXT * GROUPS + i];
            scale_raw[i * 2] = (uint8_t)value;
            scale_raw[i * 2 + 1] = (uint8_t)(value >> 8);
        }
        if(BFile_Write(fd, scale_raw, scale_bytes) != scale_bytes) goto fail;
    }
    BFile_Close(fd);
    return 1;

fail:
    BFile_Close(fd);
    return 0;
}

bool tinytalk_debug_build_prefix(void)
{
    if(!tinytalk_prepare()) return false;
    memset(&state, 0, sizeof(state));
    memcpy(state.prompt, fixed_prefix_tokens, sizeof(fixed_prefix_tokens));
    state.prompt_len = PREFIX_TOKENS;
    if(gint_world_switch(GINT_CALL(os_debug_compute_prefix, 0)) != 1)
        return false;
    if(gint_world_switch(GINT_CALL(os_debug_write_prefix, 0)) != 1) {
        set_error("Cannot write TINYTLK.PFX");
        return false;
    }
    set_error("");
    return true;
}
#endif
