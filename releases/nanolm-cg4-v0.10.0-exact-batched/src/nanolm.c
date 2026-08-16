/*
   Stateless, storage-streaming NanoLM runtime for the fx-CG50.

   Q4 weights are never copied into RAM. The calculator-specific CG4 stream
   is kept open while the add-in runs, and each projection is discarded after
   use.
   The only cache is the attention cache for the prompt currently being
   generated; nanolm_start() clears it unconditionally.
*/
#include "nanolm.h"

#include <gint/bfile.h>
#include <gint/gint.h>
#include <gint/rtc.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIDDEN 312
#define INTERMEDIATE 1092
#define HEADS 12
#define KV_HEADS 4
#define HEAD_DIM 26
#define KV_DIM (KV_HEADS * HEAD_DIM)
#define LAYERS 12
#define MAX_CONTEXT 64
#define MAX_PROMPT_TOKENS 112
#define MAX_RESPONSE_TOKENS 48
#define TENSOR_COUNT 110
#define CG4_GROUP_BYTES 36
#ifdef CASIOLLM_NANOLM_BATCH_PREFILL
#define PREFILL_BATCH 4
#endif
#ifndef IO_BUFFER_BYTES
#define IO_BUFFER_BYTES 12000
#endif

#define ID_IM_START 19996
#define ID_IM_END 19997
#define ID_EOT 19998
#define ID_NEWLINE 13
#define ID_SPACE 19855
#define ID_SYSTEM 1495
#define ID_USER 2024
#define ID_ASSISTANT 11167

#ifdef CASIOLLM_NANOLM_CG4
#ifdef CASIOLLM_NANOLM_BATCH_PREFILL
#define NANOLM_BACKEND_LOG "NanoLM-CG4-Q4-20K-B4"
#else
#define NANOLM_BACKEND_LOG "NanoLM-CG4-Q4-20K"
#endif
#else
#define NANOLM_BACKEND_LOG "NanoLM-Q4-20K"
#endif

typedef struct { uint32_t a, b, c, d; } entry_t;
typedef struct {
    entry_t index[TENSOR_COUNT];
    uint32_t data_base;
    uint32_t vocab_size;
    uint8_t model_sha256[32];
#ifdef CASIOLLM_NANOLM_HYPER
    /* These tensors never change. Converting them once removes 25 storage
       reads and 7,800 half-to-float conversions from every model token. */
    float norms[LAYERS * 2 + 1][HIDDEN];
    float rope_denominator[HEAD_DIM / 2];
    float attention_denominator;
#endif
    int model_fd;
    int index_version;
    int ready;
    char error[72];
} assets_t;

typedef struct {
    float x[HIDDEN];
    float xn[HIDDEN];
    float q[HIDDEN];
    float k[KV_DIM];
    float v[KV_DIM];
    float attn[HIDDEN];
    float gate[INTERMEDIATE];
    float up[INTERMEDIATE];
#ifdef CASIOLLM_NANOLM_HYPER
    /* One GQA KV head decoded at a time. Three query heads share it, avoiding
       two thirds of the half-float conversions in attention. */
    float key_work[MAX_CONTEXT][HEAD_DIM];
    float value_work[MAX_CONTEXT][HEAD_DIM];
#endif
    float rope_cos[HEAD_DIM / 2];
    float rope_sin[HEAD_DIM / 2];
    int16_t xq[INTERMEDIATE];
    float xq_scale;
    uint16_t key_cache[LAYERS][MAX_CONTEXT][KV_DIM];
    uint16_t value_cache[LAYERS][MAX_CONTEXT][KV_DIM];
    int prompt[MAX_PROMPT_TOKENS];
    int prompt_len;
    int prompt_at;
    int position;
    int pending;
    int have_pending;
    int generated;
    int active;
    int forward_active;
    int forward_kind;
    int forward_token;
    int forward_layer;
    int forward_stage;
    int forward_row;
    int activation_at;
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
    int cached_tokens;
    int user_tokens;
    uint32_t user_token_hash;
    int token_trace[MAX_RESPONSE_TOKENS];
    int canned_active;
    int canned_at;
    char canned_reply[96];
    char decoded[48];
} state_t;

#ifdef CASIOLLM_NANOLM_BATCH_PREFILL
/* Prompt-only batch workspace. Four causal positions share each streamed
   matrix read, while every dot product, normalization and attention operation
   retains the scalar runtime's order. At decode time this workspace is idle. */
typedef struct {
    float x[PREFILL_BATCH][HIDDEN];
    float xn[PREFILL_BATCH][HIDDEN];
    float q[PREFILL_BATCH][HIDDEN];
    float k[PREFILL_BATCH][KV_DIM];
    float v[PREFILL_BATCH][KV_DIM];
    float gate[PREFILL_BATCH][INTERMEDIATE];
    int16_t xq[PREFILL_BATCH][INTERMEDIATE];
    float xq_scale[PREFILL_BATCH];
    float rope_cos[PREFILL_BATCH][HEAD_DIM / 2];
    float rope_sin[PREFILL_BATCH][HEAD_DIM / 2];
    int active;
    int count;
    int layer;
    int stage;
} prefill_t;
#endif

static assets_t assets;
static state_t state;
#ifdef CASIOLLM_NANOLM_BATCH_PREFILL
static prefill_t prefill;
#endif
static uint8_t *io_buffer;
static char log_buffer[160];
#ifdef CASIOLLM_NANOLM_HYPER
typedef uint32_t alias_u32_t __attribute__((__may_alias__));
/* Native-endian copies of the two signed int16 values represented by each
   packed Q4 byte. Building this once turns the hot unpack loop into one table
   load and one aligned 32-bit store per pair without changing any value. */
static uint32_t q4_pair_table[256];
static int q4_pair_table_ready;
#endif

enum {
    FORWARD_PROMPT = 1,
    FORWARD_EMIT = 2,
};

enum {
    FWD_INPUT_NORM,
    FWD_Q,
    FWD_K,
    FWD_V,
    FWD_ATTENTION,
    FWD_O,
    FWD_POST_NORM,
    FWD_GATE,
    FWD_UP,
    FWD_ACTIVATION,
    FWD_DOWN,
    FWD_RESIDUAL,
    FWD_FINAL_NORM,
};

#ifdef CASIOLLM_NANOLM_SHORT_PROMPT
/* "You are CasioLLM, a helpful assistant." The longer author/behavior prompt
   was measurably worse on the 20K Q4 model. */
static int const fixed_system_text[] = {
    1834, 460, 7038, 681, 4682, 19905, 19875, 264, 9040, 11167, 19873,
};
#else
static int const fixed_system_text[] = {
    1834, 460, 7038, 681, 4682, 19905, 19875, 264, 4356, 987, 12645,
    1210, 486, 3718, 19860, 12526, 19912, 856, 19912, 12149, 301, 19873,
    3035, 346, 12338, 19873,
};
#endif

static uint32_t le32(uint8_t const *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
        ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t le16(uint8_t const *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

#ifdef CASIOLLM_NANOLM_CG4
static float be_f32(uint8_t const *p)
{
#ifdef __sh__
    typedef float alias_float_t __attribute__((__may_alias__));
    /* CG4 groups and the malloc-backed I/O buffer are 4-byte aligned, and the
       calculator is big-endian, so the stored scale already has native float
       byte order. This is the exact same bit pattern without reconstruction. */
    return *(alias_float_t const *)(void const *)p;
#else
    uint32_t bits = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
        ((uint32_t)p[2] << 8) | (uint32_t)p[3];
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}

static inline __attribute__((always_inline)) void prefetch_cg4(
    uint8_t const *p)
{
#ifdef __sh__
    __asm__ volatile("pref @%0" :: "r"(p));
#else
    (void)p;
#endif
}
#endif

#ifdef CASIOLLM_HOST_TOOLS
static void store_le16(uint8_t *p, uint16_t value)
{
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
}

static void store_le32(uint8_t *p, uint32_t value)
{
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
    p[2] = (uint8_t)(value >> 16);
    p[3] = (uint8_t)(value >> 24);
}
#endif

static uint32_t token_hash(int const *tokens, int count)
{
    uint32_t hash = 2166136261U;
    for(int i = 0; i < count; i++) {
        uint32_t token = (uint32_t)tokens[i];
        for(int byte = 0; byte < 4; byte++) {
            hash ^= (token >> (byte * 8)) & 0xffU;
            hash *= 16777619U;
        }
    }
    return hash;
}

static int fill_fixed_prefix(int *out)
{
    int n = 0;
    out[n++] = ID_IM_START;
    out[n++] = ID_SYSTEM;
    out[n++] = ID_NEWLINE;
    for(unsigned i = 0;
        i < sizeof(fixed_system_text) / sizeof(fixed_system_text[0]); i++)
        out[n++] = fixed_system_text[i];
    out[n++] = ID_IM_END;
    out[n++] = ID_SPACE;
    out[n++] = ID_NEWLINE;
    out[n++] = ID_IM_START;
    out[n++] = ID_USER;
    out[n++] = ID_NEWLINE;
    return n;
}

#ifdef CASIOLLM_NANOLM_SHORT_IDENTITY
static int select_identity_reply(char const *prompt)
{
    char normalized[80];
    int at = 0;
    int pending_space = 0;
    for(int i = 0; prompt[i] && at + 1 < (int)sizeof(normalized); i++) {
        unsigned char c = (unsigned char)prompt[i];
        if(c >= 'A' && c <= 'Z') c = (unsigned char)(c - 'A' + 'a');
        if(c >= 'a' && c <= 'z') {
            if(pending_space && at > 0) normalized[at++] = ' ';
            normalized[at++] = (char)c;
            pending_space = 0;
        }
        else if(c == ' ' || c == '\t') pending_space = 1;
    }
    normalized[at] = '\0';
    if(strcmp(normalized, "who are you") == 0 ||
       strcmp(normalized, "what are you") == 0) {
        strcpy(state.canned_reply,
            "I'm CasioLLM, a calculator AI made by Sami Gonzalez Kamel.");
    }
    else if(strcmp(normalized, "who made you") == 0 ||
            strcmp(normalized, "who created you") == 0 ||
            strcmp(normalized, "who is your creator") == 0) {
        strcpy(state.canned_reply, "I was made by Sami Gonzalez Kamel.");
    }
    else return 0;
    state.canned_active = 1;
    return 1;
}
#endif

static uint32_t elapsed_ticks(uint32_t start, uint32_t end)
{
    /* rtc_ticks() wraps at midnight. */
    uint32_t const day = 24U * 60U * 60U * 128U;
    return end >= start ? end - start : day - start + end;
}

static float f16(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 0x1f;
    uint32_t mantissa = h & 0x03ff;
    uint32_t bits;
    float value;
    if(exponent == 0) {
        if(mantissa == 0) bits = sign;
        else {
            exponent = 113;
            while((mantissa & 0x0400) == 0) { mantissa <<= 1; exponent--; }
            mantissa &= 0x03ff;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    }
    else if(exponent == 31) bits = sign | 0x7f800000 | (mantissa << 13);
    else bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint16_t to_f16(float value)
{
    uint32_t bits;
    uint32_t sign;
    uint32_t exponent;
    uint32_t mantissa;
    memcpy(&bits, &value, sizeof(bits));
    sign = (bits >> 16) & 0x8000;
    exponent = (bits >> 23) & 0xff;
    mantissa = bits & 0x7fffff;
    if(exponent <= 112) return (uint16_t)sign;
    if(exponent >= 143) return (uint16_t)(sign | 0x7c00);
    return (uint16_t)(sign | ((exponent - 112) << 10) | ((mantissa + 0x1000) >> 13));
}

static void set_error(char const *text)
{
    strncpy(assets.error, text, sizeof(assets.error) - 1);
    assets.error[sizeof(assets.error) - 1] = '\0';
}

static int os_append_log(int ignored)
{
    static uint16_t const path[] = u"\\\\fls0\\NANOLM.LOG";
    char line[192];
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
        size = 0;
    }

    length = (int)strlen(log_buffer);
    if(length > (int)sizeof(line) - 4) length = sizeof(line) - 4;
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

static void log_token_trace(void)
{
    char line[160];
    for(int begin = 0; begin < state.generated; begin += 12) {
        int end = begin + 12;
        int at;
        if(end > state.generated) end = state.generated;
        at = snprintf(line, sizeof(line), "TOKENS %d-%d ids=", begin + 1, end);
        for(int i = begin; i < end && at + 8 < (int)sizeof(line); i++)
            at += snprintf(line + at, sizeof(line) - (size_t)at,
                "%s%d", i == begin ? "" : ",", state.token_trace[i]);
        log_event(line);
    }
}

static int os_close_model(int ignored)
{
    (void)ignored;
    if(assets.model_fd >= 0) BFile_Close(assets.model_fd);
    assets.model_fd = -1;
    return 1;
}

static int os_load_assets(int ignored)
{
    uint8_t header[56];
    uint8_t raw[TENSOR_COUNT * 16];
    int fd;
    int header_size;
#ifdef CASIOLLM_NANOLM_HYPER
    uint8_t norm_raw[HIDDEN * 2];
#endif
    (void)ignored;
    assets.model_fd = -1;
    fd = BFile_Open(u"\\\\fls0\\NANOLM.IDX", BFile_ReadOnly);
    if(fd < 0) { set_error("Missing NANOLM.IDX"); return 0; }
    if(BFile_Read(fd, header, 24, 0) != 24 ||
       (memcmp(header, "NLMIDX01", 8) != 0 &&
        memcmp(header, "NLMIDX02", 8) != 0 &&
        memcmp(header, "NLMIDX03", 8) != 0) ||
       le32(header + 12) != TENSOR_COUNT ||
       le32(header + 20) != 64) {
        BFile_Close(fd); set_error("Invalid NANOLM.IDX"); return 0;
    }
    assets.index_version = header[7] == '3' ? 3 :
        (header[7] == '2' ? 2 : 1);
    header_size = assets.index_version >= 2 ? 56 : 24;
    memset(assets.model_sha256, 0, sizeof(assets.model_sha256));
    if(assets.index_version >= 2) {
        if(BFile_Read(fd, header + 24, 32, 24) != 32) {
            BFile_Close(fd); set_error("Cannot read NANOLM.IDX digest"); return 0;
        }
        memcpy(assets.model_sha256, header + 24, 32);
    }
    if(BFile_Read(fd, raw, sizeof(raw), header_size) != (int)sizeof(raw)) {
        BFile_Close(fd); set_error("Cannot read NANOLM.IDX"); return 0;
    }
    BFile_Close(fd);
    assets.data_base = le32(header + 8);
    assets.vocab_size = le32(header + 16);
    for(int i = 0; i < TENSOR_COUNT; i++) {
        assets.index[i].a = le32(raw + i * 16);
        assets.index[i].b = le32(raw + i * 16 + 4);
        assets.index[i].c = le32(raw + i * 16 + 8);
        assets.index[i].d = le32(raw + i * 16 + 12);
    }
    if(assets.vocab_size != 20000) { set_error("This build needs 20K Q4"); return 0; }
#ifdef CASIOLLM_NANOLM_HYPER
#ifdef CASIOLLM_NANOLM_CG4
    if(assets.index_version != 3) {
        set_error("CG4 needs NANOLM.IDX v3");
        return 0;
    }
    assets.model_fd = BFile_Open(u"\\\\fls0\\NANOLM.CG4", BFile_ReadOnly);
    if(assets.model_fd < 0) { set_error("Missing NANOLM.CG4"); return 0; }
    {
        uint8_t cg4_header[8];
        if(BFile_Read(assets.model_fd, cg4_header, sizeof(cg4_header), 0)
                != (int)sizeof(cg4_header) ||
           memcmp(cg4_header, "NLMCG401", 8) != 0) {
            BFile_Close(assets.model_fd);
            assets.model_fd = -1;
            set_error("Invalid NANOLM.CG4");
            return 0;
        }
    }
#else
    assets.model_fd = BFile_Open(u"\\\\fls0\\NANOLM.Q4", BFile_ReadOnly);
    if(assets.model_fd < 0) { set_error("Missing NANOLM.Q4"); return 0; }
#endif
    for(int slot = 0; slot < LAYERS * 2 + 1; slot++) {
        int tensor = slot == LAYERS * 2
            ? 1 + LAYERS * 9
            : 1 + (slot / 2) * 9 + ((slot & 1) ? 5 : 0);
        int offset = (int)(assets.data_base + assets.index[tensor].a);
        if(BFile_Read(assets.model_fd, norm_raw, sizeof(norm_raw), offset)
                != (int)sizeof(norm_raw)) {
            BFile_Close(assets.model_fd);
            assets.model_fd = -1;
            set_error("Cannot preload Nano norms");
            return 0;
        }
        for(int i = 0; i < HIDDEN; i++)
            assets.norms[slot][i] = f16(le16(norm_raw + i * 2));
    }
    for(int i = 0; i < HEAD_DIM / 2; i++)
        assets.rope_denominator[i] =
            powf(10000.0f, (float)(2 * i) / HEAD_DIM);
    assets.attention_denominator = sqrtf((float)HEAD_DIM);
#endif
    io_buffer = malloc(IO_BUFFER_BYTES);
    if(!io_buffer) {
        if(assets.model_fd >= 0) BFile_Close(assets.model_fd);
        assets.model_fd = -1;
        set_error("Not enough RAM for Q4 buffer");
        return 0;
    }
    assets.ready = 1;
    set_error("");
    return 1;
}

bool nanolm_prepare(void)
{
#ifdef CASIOLLM_NANOLM_HYPER
    if(!q4_pair_table_ready) {
        for(int packed = 0; packed < 256; packed++) {
            int16_t pair[2] = {
                (int16_t)((packed & 15) - 8),
                (int16_t)((packed >> 4) - 8),
            };
            memcpy(&q4_pair_table[packed], pair, sizeof(pair));
        }
        q4_pair_table_ready = 1;
    }
#endif
    if(assets.ready) return true;
    return gint_world_switch(GINT_CALL(os_load_assets, 0)) == 1;
}

typedef struct {
    uint32_t child, sibling;
    uint16_t token;
    uint8_t byte, pad;
    int32_t score;
} trie_node_t;

static int trie_read(int fd, uint32_t node, trie_node_t *out)
{
    uint8_t raw[16];
    if(BFile_Read(fd, raw, sizeof(raw), 12 + (int)(node * sizeof(raw)))
            != (int)sizeof(raw)) return 0;
    out->child = le32(raw);
    out->sibling = le32(raw + 4);
    out->token = le16(raw + 8);
    out->byte = raw[10];
    out->pad = raw[11];
    out->score = (int32_t)le32(raw + 12);
    return 1;
}

/* Segment only while in the OS world. The trie remains external, keeping the
   add-in below the CG50's 512 KB RAM limit. */
static int trie_find(int fd, char const *text, int length, int *token, int *score)
{
    uint32_t node = 0;
    for(int at = 0; at < length; at++) {
        trie_node_t current;
        uint32_t child;
        int found = 0;
        if(!trie_read(fd, node, &current)) return 0;
        child = current.child;
        while(child != 0xffffffffU) {
            if(!trie_read(fd, child, &current)) return 0;
            if(current.byte == (uint8_t)text[at]) { found = 1; break; }
            child = current.sibling;
        }
        if(!found) return 0;
        node = child;
    }
    {
        trie_node_t result;
        if(!trie_read(fd, node, &result) || result.token == 0xffff) return 0;
        *token = result.token;
        *score = result.score;
    }
    return 1;
}

static int tokenize_user_os(int fd, char const *text, int *out, int capacity)
{
    char normalized[256];
    struct segment { int start, length, token; } segments[128];
    int n = 0, count = 0, segment_count = 0;
    for(int i = 0; text[i] && n < (int)sizeof(normalized) - 4; i++) {
        if(text[i] == ' ') {
            normalized[n++] = (char)0xe2; normalized[n++] = (char)0x96; normalized[n++] = (char)0x81;
        } else normalized[n++] = text[i];
    }
    for(int at = 0; at < n && segment_count < 128;) {
        int length = (at + 3 <= n && (uint8_t)normalized[at] == 0xe2 &&
            (uint8_t)normalized[at + 1] == 0x96 && (uint8_t)normalized[at + 2] == 0x81) ? 3 : 1;
        int token, score;
        if(trie_find(fd, normalized + at, length, &token, &score)) {
            segments[segment_count].start = at;
            segments[segment_count].length = length;
            segments[segment_count++].token = token;
        }
        at += length;
    }
    /* SentencePiece BPE greedily applies the highest-ranked adjacent merge. */
    while(segment_count > 1) {
        int best = -1;
        int best_score = -2147483647;
        int merged_token = 0;
        for(int i = 0; i + 1 < segment_count; i++) {
            int token, score;
            int length = segments[i].length + segments[i + 1].length;
            if(trie_find(fd, normalized + segments[i].start, length, &token, &score) && score > best_score) {
                best = i;
                best_score = score;
                merged_token = token;
            }
        }
        if(best < 0) break;
        segments[best].length += segments[best + 1].length;
        segments[best].token = merged_token;
        memmove(&segments[best + 1], &segments[best + 2],
            (size_t)(segment_count - best - 2) * sizeof(segments[0]));
        segment_count--;
    }
    for(int i = 0; i < segment_count && count < capacity; i++) out[count++] = segments[i].token;
    return count;
}

static int os_append_user_prompt(char const *user_prompt)
{
    uint8_t header[12];
    int fd = BFile_Open(u"\\\\fls0\\NANOLM.TRI", BFile_ReadOnly);
    int n = state.prompt_len;
    if(fd < 0) { set_error("Missing NANOLM.TRI"); return 0; }
    if(BFile_Read(fd, header, sizeof(header), 0) != (int)sizeof(header) ||
       memcmp(header, "NLMTRE01", 8) != 0) {
        BFile_Close(fd); set_error("Invalid NANOLM.TRI"); return 0;
    }
    state.user_tokens = tokenize_user_os(fd, user_prompt, state.prompt + n,
        MAX_PROMPT_TOKENS - n - 6);
    if(user_prompt[0] && state.user_tokens == 0) {
        BFile_Close(fd);
        set_error("Tokenizer rejected prompt");
        return 0;
    }
    state.user_token_hash = token_hash(state.prompt + n, state.user_tokens);
    n += state.user_tokens;
    BFile_Close(fd);
    state.prompt[n++] = ID_IM_END; state.prompt[n++] = ID_SPACE; state.prompt[n++] = ID_NEWLINE;
    state.prompt[n++] = ID_IM_START; state.prompt[n++] = ID_ASSISTANT; state.prompt[n++] = ID_NEWLINE;
    state.prompt_len = n;
    return 1;
}

#ifdef CASIOLLM_NANOLM_EXTREME
static int os_load_prefix(int ignored)
{
    uint8_t header[64];
    int fixed_tokens[MAX_PROMPT_TOKENS];
    int prefix_len = fill_fixed_prefix(fixed_tokens);
    uint32_t expected_payload =
        (uint32_t)LAYERS * (uint32_t)prefix_len * KV_DIM * 2U * 2U;
    int fd;
    int offset = (int)sizeof(header);
    int row_bytes = prefix_len * KV_DIM * 2;
    (void)ignored;

    if(assets.index_version < 2) {
        set_error("Nano extreme needs IDX digest");
        return 0;
    }
    fd = BFile_Open(u"\\\\fls0\\NANOLM.PFX", BFile_ReadOnly);
    if(fd < 0) { set_error("Missing NANOLM.PFX"); return 0; }
    if(BFile_Read(fd, header, sizeof(header), 0) != (int)sizeof(header) ||
       memcmp(header, "NLMPFX01", 8) != 0 ||
       memcmp(header + 8, assets.model_sha256, 32) != 0 ||
       le32(header + 40) != (uint32_t)prefix_len ||
       le32(header + 44) != LAYERS || le32(header + 48) != KV_DIM ||
       le32(header + 52) != 2 || le32(header + 56) != expected_payload ||
       le32(header + 60) != token_hash(fixed_tokens, prefix_len)) {
        BFile_Close(fd);
        set_error("Invalid NANOLM.PFX");
        return 0;
    }
    for(int layer = 0; layer < LAYERS; layer++) {
        if(BFile_Read(fd, io_buffer, row_bytes, offset) != row_bytes) {
            BFile_Close(fd); set_error("Cannot read NANOLM.PFX"); return 0;
        }
        for(int position = 0; position < prefix_len; position++)
            for(int dim = 0; dim < KV_DIM; dim++) {
                int i = position * KV_DIM + dim;
                state.key_cache[layer][position][dim] =
                    le16(io_buffer + i * 2);
            }
        offset += row_bytes;
        if(BFile_Read(fd, io_buffer, row_bytes, offset) != row_bytes) {
            BFile_Close(fd); set_error("Cannot read NANOLM.PFX"); return 0;
        }
        for(int position = 0; position < prefix_len; position++)
            for(int dim = 0; dim < KV_DIM; dim++) {
                int i = position * KV_DIM + dim;
                state.value_cache[layer][position][dim] =
                    le16(io_buffer + i * 2);
            }
        offset += row_bytes;
    }
    BFile_Close(fd);
    state.position = prefix_len;
    state.cached_tokens = prefix_len;
    return 1;
}
#endif

bool nanolm_start(char const *user_prompt)
{
    char normalized_prompt[81];
    char line[160];
    size_t prompt_length;
    uint32_t request_started = rtc_ticks();

    /* Keep the backend safe when it is called outside main.c as well (host
       regression tools do this directly). Calculator-enterable prompts are
       capped at 80 bytes, so copy once and strip only trailing ASCII spaces;
       internal spaces remain semantically significant. */
    prompt_length = strlen(user_prompt);
    if(prompt_length >= sizeof(normalized_prompt))
        prompt_length = sizeof(normalized_prompt) - 1;
    memcpy(normalized_prompt, user_prompt, prompt_length);
    while(prompt_length > 0 && normalized_prompt[prompt_length - 1] == ' ')
        prompt_length--;
    normalized_prompt[prompt_length] = '\0';
    user_prompt = normalized_prompt;

    memset(&state, 0, sizeof(state));
#ifdef CASIOLLM_NANOLM_BATCH_PREFILL
    memset(&prefill, 0, sizeof(prefill));
#endif
#ifdef CASIOLLM_NANOLM_SHORT_IDENTITY
    if(select_identity_reply(user_prompt)) {
        state.active = 1;
        state.started_ticks = request_started;
        snprintf(line, sizeof(line), "START canned_identity user=%.80s", user_prompt);
        log_event(line);
        return true;
    }
#endif
    if(!nanolm_prepare()) return false;
#ifdef CASIOLLM_NANOLM_EXTREME
    if(gint_world_switch(GINT_CALL(os_load_prefix, 0)) != 1) return false;
    state.prompt_len = 0;
#else
    state.prompt_len = fill_fixed_prefix(state.prompt);
#endif
    if(gint_world_switch(GINT_CALL(os_append_user_prompt, user_prompt)) != 1) return false;
    state.active = 1;
    state.started_ticks = request_started;
    snprintf(line, sizeof(line),
        "START backend=%s p=%d cached=%d utok=%d uhash=%08lx user=%.46s",
        NANOLM_BACKEND_LOG, state.prompt_len, state.cached_tokens, state.user_tokens,
        (unsigned long)state.user_token_hash, user_prompt);
    log_event(line);
    return true;
}

static int read_at(int fd, void *buffer, int bytes, uint32_t offset)
{
    return BFile_Read(fd, buffer, bytes, (int)(assets.data_base + offset)) == bytes;
}

static void __attribute__((unused)) rms_norm_f16(
    float *out, float const *x, uint8_t const *weights)
{
    float sum = 0.0f;
    for(int i = 0; i < HIDDEN; i++) sum += x[i] * x[i];
    sum = 1.0f / sqrtf(sum / HIDDEN + 1e-6f);
    for(int i = 0; i < HIDDEN; i++) out[i] = x[i] * sum * f16(le16(weights + i * 2));
}

#ifdef CASIOLLM_NANOLM_HYPER
static void rms_norm_cached(float *out, float const *x, float const *weights)
{
    float sum = 0.0f;
    for(int i = 0; i < HIDDEN; i++) sum += x[i] * x[i];
    sum = 1.0f / sqrtf(sum / HIDDEN + 1e-6f);
    for(int i = 0; i < HIDDEN; i++) out[i] = x[i] * sum * weights[i];
}
#endif

static float quantize_activation(float const *x, int count, int16_t *out)
{
    float maximum = 0.0f;
    float inverse;
    float scale;

    for(int i = 0; i < count; i++) {
        float absolute = fabsf(x[i]);
        if(absolute > maximum) maximum = absolute;
    }
    if(maximum < 1e-20f) {
        memset(out, 0, (size_t)count * sizeof(*out));
        return 0.0f;
    }
    scale = maximum / 32767.0f;
    inverse = 32767.0f / maximum;
    for(int i = 0; i < count; i++) {
        float scaled = x[i] * inverse;
        int value = (int)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
        if(value > 32767) value = 32767;
        if(value < -32767) value = -32767;
        out[i] = (int16_t)value;
    }
    return scale;
}

#ifdef CASIOLLM_NANOLM_HYPER
/* The SH4 MAC.W instruction performs a signed 16x16 multiply and accumulates
   it without the per-element mul.l/sts sequence emitted by the generic C
   loop. Q4 values are unpacked to a 64-value scratch row before this call.
   The accumulator cannot overflow for q in [-7, 7] and INT16 activations. */
static inline __attribute__((always_inline)) int32_t dot_i16_exact(
    int16_t const *weights, int16_t const *activation, int count)
{
#ifdef __sh__
    int32_t total;
    int16_t const *w = weights;
    int16_t const *x = activation;
    int blocks = count >> 3;
    int tail = count & 7;
    __asm__ volatile(
        "clrmac\n\t"
        "tst %[blocks],%[blocks]\n\t"
        "bt 2f\n\t"
        "1:\n\t"
        "mac.w @%[w]+,@%[x]+\n\t"
        "mac.w @%[w]+,@%[x]+\n\t"
        "mac.w @%[w]+,@%[x]+\n\t"
        "mac.w @%[w]+,@%[x]+\n\t"
        "mac.w @%[w]+,@%[x]+\n\t"
        "mac.w @%[w]+,@%[x]+\n\t"
        "mac.w @%[w]+,@%[x]+\n\t"
        "mac.w @%[w]+,@%[x]+\n\t"
        "dt %[blocks]\n\t"
        "bf 1b\n\t"
        "2:\n\t"
        "tst %[tail],%[tail]\n\t"
        "bt 4f\n\t"
        "3:\n\t"
        "mac.w @%[w]+,@%[x]+\n\t"
        "dt %[tail]\n\t"
        "bf 3b\n\t"
        "4:\n\t"
        "sts macl,%[total]"
        : [w] "+r"(w), [x] "+r"(x), [blocks] "+r"(blocks),
          [tail] "+r"(tail),
          [total] "=&r"(total)
        :
        : "mach", "macl", "memory");
    return total;
#else
    int32_t total = 0;
    for(int i = 0; i < count; i++)
        total += (int32_t)weights[i] * activation[i];
    return total;
#endif
}

static inline __attribute__((always_inline)) int32_t dot_q4_i16_exact(
    uint8_t const *quant, int quant_first_element, int element,
    int16_t const *activation, int count, int16_t *unpacked)
{
    int at = 0;
    if((element & 1) == 0) {
        for(; at + 1 < count; at += 2) {
            uint8_t packed = quant[((element + at) >> 1) - quant_first_element / 2];
            *(alias_u32_t *)(void *)(unpacked + at) = q4_pair_table[packed];
        }
    }
    for(; at < count; at++) {
        int current = element + at;
        uint8_t packed = quant[(current >> 1) - quant_first_element / 2];
        unpacked[at] = (int16_t)(((current & 1)
            ? (packed >> 4) : (packed & 15)) - 8);
    }
    return dot_i16_exact(unpacked, activation, count);
}

#ifdef CASIOLLM_NANOLM_CG4
static inline __attribute__((always_inline)) void unpack_cg4_exact(
    uint8_t const *quant, int group_offset, int count, int16_t *unpacked)
{
    int at = 0;
    if((group_offset & 1) == 0) {
        for(; at + 1 < count; at += 2) {
            uint8_t packed = quant[(group_offset + at) >> 1];
            *(alias_u32_t *)(void *)(unpacked + at) = q4_pair_table[packed];
        }
    }
    for(; at < count; at++) {
        int current = group_offset + at;
        uint8_t packed = quant[current >> 1];
        unpacked[at] = (int16_t)(((current & 1)
            ? (packed >> 4) : (packed & 15)) - 8);
    }
}

static inline __attribute__((always_inline)) int32_t dot_cg4_i16_exact(
    uint8_t const *quant, int group_offset, int16_t const *activation,
    int count, int16_t *unpacked)
{
    unpack_cg4_exact(quant, group_offset, count, unpacked);
    return dot_i16_exact(unpacked, activation, count);
}
#endif
#endif

/* Process up to 16 storage blocks of a Q4 matrix. Activations are INT16, so
   the inner loop is entirely integer; floating point is used only once per
   64-weight quantization group. Returns -1 on I/O failure, 0 while more rows
   remain, and 1 when the matrix is complete. */
static int matvec_microstep(int fd, entry_t const *entry, int rows, int cols,
    int16_t const *x, float x_scale, float *out, int *begin_ptr)
{
    int begin = *begin_ptr;
#ifdef CASIOLLM_NANOLM_HYPER
    int16_t unpacked[64];
#endif

    for(int block = 0; block < 16 && begin < rows; block++) {
        int count = rows - begin;
        int flat = begin * cols;
        int first_group;
        int groups;
#ifdef CASIOLLM_NANOLM_CG4
        int stream_bytes;
#else
        int scale_bytes, quant_bytes;
        uint8_t *scales, *quant;
#endif

        while(count > 1) {
            first_group = flat >> 6;
            groups = ((flat + count * cols - 1) >> 6) - first_group + 1;
#ifdef CASIOLLM_NANOLM_CG4
            stream_bytes = groups * CG4_GROUP_BYTES;
            if(stream_bytes <= IO_BUFFER_BYTES) break;
#else
            scale_bytes = groups * 2;
            quant_bytes = count * cols / 2;
            if(scale_bytes + quant_bytes <= IO_BUFFER_BYTES) break;
#endif
            count--;
        }
        first_group = flat >> 6;
        groups = ((flat + count * cols - 1) >> 6) - first_group + 1;
#ifdef CASIOLLM_NANOLM_CG4
        stream_bytes = groups * CG4_GROUP_BYTES;
        if(stream_bytes > IO_BUFFER_BYTES ||
           !read_at(fd, io_buffer, stream_bytes,
               entry->a + (uint32_t)first_group * CG4_GROUP_BYTES)) return -1;
#else
        scale_bytes = groups * 2;
        quant_bytes = count * cols / 2;
        if(scale_bytes + quant_bytes > IO_BUFFER_BYTES) return -1;
        scales = io_buffer;
        quant = io_buffer + scale_bytes;
        if(!read_at(fd, scales, scale_bytes, entry->a + first_group * 2) ||
           !read_at(fd, quant, quant_bytes, entry->b + flat / 2)) return -1;
#endif

        for(int local = 0; local < count; local++) {
            int row = begin + local;
            int col = 0;
            float total = 0.0f;
            while(col < cols) {
                int element = row * cols + col;
                int group = element >> 6;
                int take = 64 - (element & 63);
                int32_t subtotal = 0;
                if(take > cols - col) take = cols - col;
#ifdef CASIOLLM_NANOLM_HYPER
#ifdef CASIOLLM_NANOLM_CG4
                {
                    uint8_t const *group_data = io_buffer +
                        (group - first_group) * CG4_GROUP_BYTES;
                    if(group - first_group + 1 < groups)
                        prefetch_cg4(group_data + CG4_GROUP_BYTES);
                    subtotal = dot_cg4_i16_exact(group_data + 4,
                        element & 63, x + col, take, unpacked);
                    total += (float)subtotal * be_f32(group_data);
                }
#else
                subtotal = dot_q4_i16_exact(quant, flat, element,
                    x + col, take, unpacked);
#endif
#else
                for(int j = 0; j < take; j++) {
                    int current = element + j;
                    uint8_t packed = quant[(current >> 1) - flat / 2];
                    int q = ((current & 1) ? (packed >> 4) : (packed & 15)) - 8;
                    subtotal += q * (int32_t)x[col + j];
                }
#endif
#ifndef CASIOLLM_NANOLM_CG4
                total += (float)subtotal *
                    f16(le16(scales + (group - first_group) * 2));
#endif
                col += take;
            }
            out[row] = total * x_scale;
        }
        begin += count;
    }

    *begin_ptr = begin;
    return begin >= rows ? 1 : 0;
}

#if defined(CASIOLLM_NANOLM_BATCH_PREFILL) && defined(CASIOLLM_NANOLM_CG4)
/* Apply one streamed matrix to every prompt position in the current batch.
   The packed weights are read and unpacked once, but each scalar dot product
   and each floating-point accumulation keeps the exact single-token order. */
static int matvec_prefill_batch(int fd, entry_t const *entry, int rows, int cols,
    float *out, int out_stride, int fuse_silu_up)
{
    int begin = 0;
    int16_t unpacked[64];

    while(begin < rows) {
        int count = rows - begin;
        int flat = begin * cols;
        int first_group;
        int groups;
        int stream_bytes;

        while(count > 1) {
            first_group = flat >> 6;
            groups = ((flat + count * cols - 1) >> 6) - first_group + 1;
            stream_bytes = groups * CG4_GROUP_BYTES;
            if(stream_bytes <= IO_BUFFER_BYTES) break;
            count--;
        }
        first_group = flat >> 6;
        groups = ((flat + count * cols - 1) >> 6) - first_group + 1;
        stream_bytes = groups * CG4_GROUP_BYTES;
        if(stream_bytes > IO_BUFFER_BYTES ||
           !read_at(fd, io_buffer, stream_bytes,
               entry->a + (uint32_t)first_group * CG4_GROUP_BYTES)) return 0;

        for(int local = 0; local < count; local++) {
            int row = begin + local;
            int col = 0;
            float totals[PREFILL_BATCH] = { 0.0f };
            while(col < cols) {
                int element = row * cols + col;
                int group = element >> 6;
                int take = 64 - (element & 63);
                uint8_t const *group_data;
                float weight_scale;
                if(take > cols - col) take = cols - col;
                group_data = io_buffer +
                    (group - first_group) * CG4_GROUP_BYTES;
                if(group - first_group + 1 < groups)
                    prefetch_cg4(group_data + CG4_GROUP_BYTES);
                weight_scale = be_f32(group_data);
                unpack_cg4_exact(group_data + 4, element & 63, take, unpacked);
                for(int batch = 0; batch < prefill.count; batch++) {
                    int32_t subtotal = dot_i16_exact(unpacked,
                        prefill.xq[batch] + col, take);
                    totals[batch] += (float)subtotal * weight_scale;
                }
                col += take;
            }
            for(int batch = 0; batch < prefill.count; batch++) {
                float value = totals[batch] * prefill.xq_scale[batch];
                float *destination = out + batch * out_stride + row;
                if(fuse_silu_up) {
                    float gate = *destination;
                    *destination = gate / (1.0f + expf(-gate)) * value;
                }
                else *destination = value;
            }
        }
        begin += count;
    }
    return 1;
}
#endif

static int embedding(int fd, int token, float *out)
{
    entry_t const *entry = &assets.index[0];
    int start = token * HIDDEN;
    int first_group = start >> 6;
    int groups = ((start + HIDDEN - 1) >> 6) - first_group + 1;
#ifdef CASIOLLM_NANOLM_CG4
    uint8_t groups_data[6 * CG4_GROUP_BYTES];
#else
    uint8_t scales[12];
    uint8_t quant[160];
    int first_byte = start >> 1;
#endif
#ifdef CASIOLLM_NANOLM_CG4
    if(token < 0 || token >= 20000 ||
       !read_at(fd, groups_data, groups * CG4_GROUP_BYTES,
           entry->a + (uint32_t)first_group * CG4_GROUP_BYTES)) return 0;
#else
    if(token < 0 || token >= 20000 ||
       !read_at(fd, scales, groups * 2, entry->a + first_group * 2) ||
       !read_at(fd, quant, 156 + (start & 1), entry->b + first_byte)) return 0;
#endif
    for(int i = 0; i < HIDDEN; i++) {
        int flat = start + i;
#ifdef CASIOLLM_NANOLM_CG4
        uint8_t const *group_data = groups_data +
            ((flat >> 6) - first_group) * CG4_GROUP_BYTES;
        int group_offset = flat & 63;
        uint8_t packed = group_data[4 + (group_offset >> 1)];
        int q = ((group_offset & 1) ? (packed >> 4) : (packed & 15)) - 8;
        out[i] = (float)q * be_f32(group_data);
#else
        uint8_t packed = quant[(flat >> 1) - first_byte];
        int q = ((flat & 1) ? (packed >> 4) : (packed & 15)) - 8;
        out[i] = (float)q * f16(le16(scales + ((flat >> 6) - first_group) * 2));
#endif
    }
    return 1;
}

static void prepare_rope_to(int position, float *cos_values, float *sin_values)
{
    for(int i = 0; i < HEAD_DIM / 2; i++) {
#ifdef CASIOLLM_NANOLM_HYPER
        float theta = (float)position / assets.rope_denominator[i];
#else
        float theta = (float)position /
            powf(10000.0f, (float)(2 * i) / HEAD_DIM);
#endif
        cos_values[i] = cosf(theta);
        sin_values[i] = sinf(theta);
    }
}

static void prepare_rope(int position)
{
    prepare_rope_to(position, state.rope_cos, state.rope_sin);
}

static void apply_rope_values(float *vector, int heads,
    float const *cos_values, float const *sin_values)
{
    for(int i = 0; i < HEAD_DIM / 2; i++) {
        float c = cos_values[i];
        float s = sin_values[i];
        for(int h = 0; h < heads; h++) {
            float *v = vector + h * HEAD_DIM;
            float a = v[i], b = v[i + HEAD_DIM / 2];
            v[i] = a * c - b * s;
            v[i + HEAD_DIM / 2] = b * c + a * s;
        }
    }
}

static void apply_rope(float *vector, int heads)
{
    apply_rope_values(vector, heads, state.rope_cos, state.rope_sin);
}

#ifdef CASIOLLM_NANOLM_HYPER
static void attention_hyper_compute(int layer, int position,
    float const *query_values, float *output)
{
    float scores[MAX_CONTEXT];
    for(int kv_head = 0; kv_head < KV_HEADS; kv_head++) {
        for(int t = 0; t <= position; t++) {
            for(int d = 0; d < HEAD_DIM; d++) {
                int at = kv_head * HEAD_DIM + d;
                state.key_work[t][d] =
                    f16(state.key_cache[layer][t][at]);
                state.value_work[t][d] =
                    f16(state.value_cache[layer][t][at]);
            }
        }
        for(int query = 0; query < HEADS / KV_HEADS; query++) {
            int head = kv_head * (HEADS / KV_HEADS) + query;
            float max_score = -1e30f, denom = 0.0f;
            for(int t = 0; t <= position; t++) {
                float sum = 0.0f;
                for(int d = 0; d < HEAD_DIM; d++)
                    sum += query_values[head * HEAD_DIM + d] *
                        state.key_work[t][d];
                scores[t] = sum / assets.attention_denominator;
                if(scores[t] > max_score) max_score = scores[t];
            }
            for(int t = 0; t <= position; t++) {
                scores[t] = expf(scores[t] - max_score);
                denom += scores[t];
            }
            for(int t = 0; t <= position; t++) scores[t] /= denom;
            for(int d = 0; d < HEAD_DIM; d++) {
                float sum = 0.0f;
                for(int t = 0; t <= position; t++)
                    sum += scores[t] * state.value_work[t][d];
                output[head * HEAD_DIM + d] = sum;
            }
        }
    }
}
#endif

static int forward_begin(int fd, int token, int kind)
{
    if(state.position >= MAX_CONTEXT) {
        set_error("Context limit reached");
        return 0;
    }
    if(!embedding(fd, token, state.x)) return 0;
    prepare_rope(state.position);
    state.forward_active = 1;
    state.forward_kind = kind;
    state.forward_token = token;
    state.forward_layer = 0;
    state.forward_stage = FWD_INPUT_NORM;
    state.forward_row = 0;
    state.activation_at = 0;
    return 1;
}

/* Advance a forward pass by one bounded unit. This returns control to gint
   frequently enough for the UI animation and F6 cancellation to work. */
static int forward_microstep(int fd)
{
#ifndef CASIOLLM_NANOLM_HYPER
    uint8_t norms[HIDDEN * 2];
    float scores[MAX_CONTEXT];
#endif
    int layer = state.forward_layer;
    int index = 1 + layer * 9;
    int result;

    if(!state.forward_active) return 1;
    if(state.forward_stage == FWD_FINAL_NORM) {
#ifdef CASIOLLM_NANOLM_HYPER
        rms_norm_cached(state.x, state.x, assets.norms[LAYERS * 2]);
#else
        if(!read_at(fd, norms, HIDDEN * 2, assets.index[1 + LAYERS * 9].a))
            return -1;
        rms_norm_f16(state.x, state.x, norms);
#endif
        state.position++;
        state.forward_active = 0;
        return 1;
    }

    switch(state.forward_stage) {
        case FWD_INPUT_NORM:
#ifdef CASIOLLM_NANOLM_HYPER
            rms_norm_cached(state.xn, state.x,
                assets.norms[state.forward_layer * 2]);
#else
            if(!read_at(fd, norms, HIDDEN * 2, assets.index[index].a)) return -1;
            rms_norm_f16(state.xn, state.x, norms);
#endif
            state.xq_scale = quantize_activation(state.xn, HIDDEN, state.xq);
            state.forward_row = 0;
            state.forward_stage = FWD_Q;
            return 0;

        case FWD_Q:
            result = matvec_microstep(fd, &assets.index[index + 1], HIDDEN,
                HIDDEN, state.xq, state.xq_scale, state.q, &state.forward_row);
            if(result < 0) return -1;
            if(result > 0) { state.forward_row = 0; state.forward_stage = FWD_K; }
            return 0;

        case FWD_K:
            result = matvec_microstep(fd, &assets.index[index + 2], KV_DIM,
                HIDDEN, state.xq, state.xq_scale, state.k, &state.forward_row);
            if(result < 0) return -1;
            if(result > 0) { state.forward_row = 0; state.forward_stage = FWD_V; }
            return 0;

        case FWD_V:
            result = matvec_microstep(fd, &assets.index[index + 3], KV_DIM,
                HIDDEN, state.xq, state.xq_scale, state.v, &state.forward_row);
            if(result < 0) return -1;
            if(result > 0) state.forward_stage = FWD_ATTENTION;
            return 0;

        case FWD_ATTENTION:
            apply_rope(state.q, HEADS);
            apply_rope(state.k, KV_HEADS);
            for(int i = 0; i < KV_DIM; i++) {
                state.key_cache[layer][state.position][i] = to_f16(state.k[i]);
                state.value_cache[layer][state.position][i] = to_f16(state.v[i]);
            }
#ifdef CASIOLLM_NANOLM_HYPER
            attention_hyper_compute(layer, state.position, state.q, state.attn);
#else
            for(int head = 0; head < HEADS; head++) {
                int kv_head = head / (HEADS / KV_HEADS);
                float max_score = -1e30f, denom = 0.0f;
                for(int t = 0; t <= state.position; t++) {
                    float sum = 0.0f;
                    for(int d = 0; d < HEAD_DIM; d++)
                        sum += state.q[head * HEAD_DIM + d] *
                            f16(state.key_cache[layer][t][kv_head * HEAD_DIM + d]);
                    scores[t] = sum / sqrtf((float)HEAD_DIM);
                    if(scores[t] > max_score) max_score = scores[t];
                }
                for(int t = 0; t <= state.position; t++) {
                    scores[t] = expf(scores[t] - max_score);
                    denom += scores[t];
                }
                for(int d = 0; d < HEAD_DIM; d++) {
                    float sum = 0.0f;
                    for(int t = 0; t <= state.position; t++)
                        sum += scores[t] / denom *
                            f16(state.value_cache[layer][t][kv_head * HEAD_DIM + d]);
                    state.attn[head * HEAD_DIM + d] = sum;
                }
            }
#endif
            state.xq_scale = quantize_activation(state.attn, HIDDEN, state.xq);
            state.forward_row = 0;
            state.forward_stage = FWD_O;
            return 0;

        case FWD_O:
            result = matvec_microstep(fd, &assets.index[index + 4], HIDDEN,
                HIDDEN, state.xq, state.xq_scale, state.xn, &state.forward_row);
            if(result < 0) return -1;
            if(result > 0) state.forward_stage = FWD_POST_NORM;
            return 0;

        case FWD_POST_NORM:
            for(int i = 0; i < HIDDEN; i++) state.x[i] += state.xn[i];
#ifdef CASIOLLM_NANOLM_HYPER
            rms_norm_cached(state.xn, state.x,
                assets.norms[state.forward_layer * 2 + 1]);
#else
            if(!read_at(fd, norms, HIDDEN * 2, assets.index[index + 5].a)) return -1;
            rms_norm_f16(state.xn, state.x, norms);
#endif
            state.xq_scale = quantize_activation(state.xn, HIDDEN, state.xq);
            state.forward_row = 0;
            state.forward_stage = FWD_GATE;
            return 0;

        case FWD_GATE:
            result = matvec_microstep(fd, &assets.index[index + 6], INTERMEDIATE,
                HIDDEN, state.xq, state.xq_scale, state.gate, &state.forward_row);
            if(result < 0) return -1;
            if(result > 0) { state.forward_row = 0; state.forward_stage = FWD_UP; }
            return 0;

        case FWD_UP:
            result = matvec_microstep(fd, &assets.index[index + 7], INTERMEDIATE,
                HIDDEN, state.xq, state.xq_scale, state.up, &state.forward_row);
            if(result < 0) return -1;
            if(result > 0) { state.activation_at = 0; state.forward_stage = FWD_ACTIVATION; }
            return 0;

        case FWD_ACTIVATION: {
#ifdef CASIOLLM_NANOLM_HYPER
            int end = INTERMEDIATE;
#else
            int end = state.activation_at + 64;
            if(end > INTERMEDIATE) end = INTERMEDIATE;
#endif
            for(int i = state.activation_at; i < end; i++)
                state.gate[i] = state.gate[i] /
                    (1.0f + expf(-state.gate[i])) * state.up[i];
            state.activation_at = end;
            if(end == INTERMEDIATE) {
                state.xq_scale = quantize_activation(state.gate, INTERMEDIATE, state.xq);
                state.forward_row = 0;
                state.forward_stage = FWD_DOWN;
            }
            return 0;
        }

        case FWD_DOWN:
            result = matvec_microstep(fd, &assets.index[index + 8], HIDDEN,
                INTERMEDIATE, state.xq, state.xq_scale, state.xn, &state.forward_row);
            if(result < 0) return -1;
            if(result > 0) state.forward_stage = FWD_RESIDUAL;
            return 0;

        case FWD_RESIDUAL:
            for(int i = 0; i < HIDDEN; i++) state.x[i] += state.xn[i];
            state.forward_layer++;
            state.forward_row = 0;
            state.forward_stage = state.forward_layer == LAYERS
                ? FWD_FINAL_NORM : FWD_INPUT_NORM;
            return 0;

        default:
            return -1;
    }
}

#if defined(CASIOLLM_NANOLM_BATCH_PREFILL) && defined(CASIOLLM_NANOLM_CG4)
static int prefill_begin(int fd)
{
    int remaining = state.prompt_len - state.prompt_at;
    int capacity = MAX_CONTEXT - state.position;
    if(remaining <= 0) return 1;
    if(capacity <= 0) {
        set_error("Context limit reached");
        return 0;
    }
    prefill.count = remaining;
    if(prefill.count > PREFILL_BATCH) prefill.count = PREFILL_BATCH;
    if(prefill.count > capacity) prefill.count = capacity;
    for(int batch = 0; batch < prefill.count; batch++) {
        int token = state.prompt[state.prompt_at + batch];
        if(!embedding(fd, token, prefill.x[batch])) return 0;
        prepare_rope_to(state.position + batch,
            prefill.rope_cos[batch], prefill.rope_sin[batch]);
    }
    prefill.layer = 0;
    prefill.stage = FWD_INPUT_NORM;
    prefill.active = 1;
    return 1;
}

/* One exact causal-prefill stage. Positions are processed together only for
   matrix streaming; attention is still committed in causal order so the KV
   state is identical to evaluating the same tokens one at a time. */
static int prefill_microstep(int fd)
{
    int layer = prefill.layer;
    int index = 1 + layer * 9;

    switch(prefill.stage) {
        case FWD_INPUT_NORM:
            for(int batch = 0; batch < prefill.count; batch++) {
                rms_norm_cached(prefill.xn[batch], prefill.x[batch],
                    assets.norms[layer * 2]);
                prefill.xq_scale[batch] = quantize_activation(
                    prefill.xn[batch], HIDDEN, prefill.xq[batch]);
            }
            prefill.stage = FWD_Q;
            return 0;

        case FWD_Q:
            if(!matvec_prefill_batch(fd, &assets.index[index + 1], HIDDEN,
                HIDDEN, &prefill.q[0][0], HIDDEN, 0)) return -1;
            prefill.stage = FWD_K;
            return 0;

        case FWD_K:
            if(!matvec_prefill_batch(fd, &assets.index[index + 2], KV_DIM,
                HIDDEN, &prefill.k[0][0], KV_DIM, 0)) return -1;
            prefill.stage = FWD_V;
            return 0;

        case FWD_V:
            if(!matvec_prefill_batch(fd, &assets.index[index + 3], KV_DIM,
                HIDDEN, &prefill.v[0][0], KV_DIM, 0)) return -1;
            prefill.stage = FWD_ATTENTION;
            return 0;

        case FWD_ATTENTION:
            for(int batch = 0; batch < prefill.count; batch++) {
                int position = state.position + batch;
                apply_rope_values(prefill.q[batch], HEADS,
                    prefill.rope_cos[batch], prefill.rope_sin[batch]);
                apply_rope_values(prefill.k[batch], KV_HEADS,
                    prefill.rope_cos[batch], prefill.rope_sin[batch]);
                for(int i = 0; i < KV_DIM; i++) {
                    state.key_cache[layer][position][i] =
                        to_f16(prefill.k[batch][i]);
                    state.value_cache[layer][position][i] =
                        to_f16(prefill.v[batch][i]);
                }
                attention_hyper_compute(layer, position,
                    prefill.q[batch], prefill.q[batch]);
                prefill.xq_scale[batch] = quantize_activation(
                    prefill.q[batch], HIDDEN, prefill.xq[batch]);
            }
            prefill.stage = FWD_O;
            return 0;

        case FWD_O:
            if(!matvec_prefill_batch(fd, &assets.index[index + 4], HIDDEN,
                HIDDEN, &prefill.xn[0][0], HIDDEN, 0)) return -1;
            prefill.stage = FWD_POST_NORM;
            return 0;

        case FWD_POST_NORM:
            for(int batch = 0; batch < prefill.count; batch++) {
                for(int i = 0; i < HIDDEN; i++)
                    prefill.x[batch][i] += prefill.xn[batch][i];
                rms_norm_cached(prefill.xn[batch], prefill.x[batch],
                    assets.norms[layer * 2 + 1]);
                prefill.xq_scale[batch] = quantize_activation(
                    prefill.xn[batch], HIDDEN, prefill.xq[batch]);
            }
            prefill.stage = FWD_GATE;
            return 0;

        case FWD_GATE:
            if(!matvec_prefill_batch(fd, &assets.index[index + 6], INTERMEDIATE,
                HIDDEN, &prefill.gate[0][0], INTERMEDIATE, 0)) return -1;
            prefill.stage = FWD_UP;
            return 0;

        case FWD_UP:
            if(!matvec_prefill_batch(fd, &assets.index[index + 7], INTERMEDIATE,
                HIDDEN, &prefill.gate[0][0], INTERMEDIATE, 1)) return -1;
            for(int batch = 0; batch < prefill.count; batch++)
                prefill.xq_scale[batch] = quantize_activation(
                    prefill.gate[batch], INTERMEDIATE, prefill.xq[batch]);
            prefill.stage = FWD_DOWN;
            return 0;

        case FWD_DOWN:
            if(!matvec_prefill_batch(fd, &assets.index[index + 8], HIDDEN,
                INTERMEDIATE, &prefill.xn[0][0], HIDDEN, 0)) return -1;
            prefill.stage = FWD_RESIDUAL;
            return 0;

        case FWD_RESIDUAL:
            for(int batch = 0; batch < prefill.count; batch++)
                for(int i = 0; i < HIDDEN; i++)
                    prefill.x[batch][i] += prefill.xn[batch][i];
            prefill.layer++;
            prefill.stage = prefill.layer == LAYERS
                ? FWD_FINAL_NORM : FWD_INPUT_NORM;
            return 0;

        case FWD_FINAL_NORM:
            for(int batch = 0; batch < prefill.count; batch++)
                rms_norm_cached(prefill.x[batch], prefill.x[batch],
                    assets.norms[LAYERS * 2]);
            memcpy(state.x, prefill.x[prefill.count - 1], sizeof(state.x));
            state.position += prefill.count;
            state.prompt_at += prefill.count;
            prefill.active = 0;
            return 1;

        default:
            return -1;
    }
}
#endif

#ifdef CASIOLLM_HOST_TOOLS
bool nanolm_debug_build_prefix(void)
{
    uint8_t header[64];
    int fixed_tokens[MAX_PROMPT_TOKENS];
    int prefix_len = fill_fixed_prefix(fixed_tokens);
    int row_bytes = prefix_len * KV_DIM * 2;
    uint32_t payload = (uint32_t)LAYERS * (uint32_t)row_bytes * 2U;
    int model_fd;
    int output_fd;
    int empty_size = 0;

    if(!nanolm_prepare()) return false;
    if(assets.index_version != 2) {
        set_error("Prefix builder needs IDX v2");
        return false;
    }
    memset(&state, 0, sizeof(state));
    model_fd = BFile_Open(u"\\\\fls0\\NANOLM.Q4", BFile_ReadOnly);
    if(model_fd < 0) { set_error("Missing NANOLM.Q4"); return false; }
    for(int at = 0; at < prefix_len; at++) {
        if(!forward_begin(model_fd, fixed_tokens[at], FORWARD_PROMPT)) {
            BFile_Close(model_fd);
            set_error("Prefix forward start failed");
            return false;
        }
        while(state.forward_active) {
            if(forward_microstep(model_fd) < 0) {
                BFile_Close(model_fd);
                set_error("Prefix forward failed");
                return false;
            }
        }
    }
    BFile_Close(model_fd);

    memset(header, 0, sizeof(header));
    memcpy(header, "NLMPFX01", 8);
    memcpy(header + 8, assets.model_sha256, 32);
    store_le32(header + 40, (uint32_t)prefix_len);
    store_le32(header + 44, LAYERS);
    store_le32(header + 48, KV_DIM);
    store_le32(header + 52, 2);
    store_le32(header + 56, payload);
    store_le32(header + 60, token_hash(fixed_tokens, prefix_len));

    (void)BFile_Remove(u"\\\\fls0\\NANOLM.PFX");
    if(BFile_Create(u"\\\\fls0\\NANOLM.PFX", BFile_File, &empty_size) < 0) {
        set_error("Cannot create NANOLM.PFX");
        return false;
    }
    output_fd = BFile_Open(u"\\\\fls0\\NANOLM.PFX", BFile_ReadWrite);
    if(output_fd < 0 ||
       BFile_Write(output_fd, header, sizeof(header)) != (int)sizeof(header)) {
        if(output_fd >= 0) BFile_Close(output_fd);
        set_error("Cannot write NANOLM.PFX");
        return false;
    }
    for(int layer = 0; layer < LAYERS; layer++) {
        for(int position = 0; position < prefix_len; position++)
            for(int dim = 0; dim < KV_DIM; dim++) {
                int i = position * KV_DIM + dim;
                store_le16(io_buffer + i * 2,
                    state.key_cache[layer][position][dim]);
            }
        if(BFile_Write(output_fd, io_buffer, row_bytes) != row_bytes) {
            BFile_Close(output_fd);
            set_error("Cannot write NANOLM.PFX payload");
            return false;
        }
        for(int position = 0; position < prefix_len; position++)
            for(int dim = 0; dim < KV_DIM; dim++) {
                int i = position * KV_DIM + dim;
                store_le16(io_buffer + i * 2,
                    state.value_cache[layer][position][dim]);
            }
        if(BFile_Write(output_fd, io_buffer, row_bytes) != row_bytes) {
            BFile_Close(output_fd);
            set_error("Cannot write NANOLM.PFX payload");
            return false;
        }
    }
    BFile_Close(output_fd);
    set_error("");
    return true;
}
#endif

static void choose_token_begin(void)
{
    state.xq_scale = quantize_activation(state.x, HIDDEN, state.xq);
    state.choosing = 1;
    state.choose_at = 0;
    state.choose_best_id = 0;
    state.choose_best = -1e30f;
}

/* Scan at most 16 embedding-table blocks. The activation scale is common to
   every vocabulary row and therefore does not affect the argmax. */
static int choose_token_microstep(int fd)
{
    entry_t const *entry = &assets.index[0];
#ifdef CASIOLLM_NANOLM_HYPER
    int16_t unpacked[64];
#endif

    for(int block = 0; block < 16 && state.choose_at < 20000; block++) {
        int begin = state.choose_at;
        int rows = 20000 - begin;
        int flat = begin * HIDDEN;
        int first_group;
        int groups;
#ifdef CASIOLLM_NANOLM_CG4
        int stream_bytes;
#else
        int scale_bytes, quant_bytes;
        uint8_t *scales, *quant;
#endif

        while(rows > 1) {
            first_group = flat >> 6;
            groups = ((flat + rows * HIDDEN - 1) >> 6) - first_group + 1;
#ifdef CASIOLLM_NANOLM_CG4
            stream_bytes = groups * CG4_GROUP_BYTES;
            if(stream_bytes <= IO_BUFFER_BYTES) break;
#else
            scale_bytes = groups * 2;
            quant_bytes = rows * HIDDEN / 2;
            if(scale_bytes + quant_bytes <= IO_BUFFER_BYTES) break;
#endif
            rows--;
        }
        first_group = flat >> 6;
        groups = ((flat + rows * HIDDEN - 1) >> 6) - first_group + 1;
#ifdef CASIOLLM_NANOLM_CG4
        stream_bytes = groups * CG4_GROUP_BYTES;
        if(stream_bytes > IO_BUFFER_BYTES ||
           !read_at(fd, io_buffer, stream_bytes,
               entry->a + (uint32_t)first_group * CG4_GROUP_BYTES)) return -1;
#else
        scale_bytes = groups * 2;
        quant_bytes = rows * HIDDEN / 2;
        if(scale_bytes + quant_bytes > IO_BUFFER_BYTES) return -1;
        scales = io_buffer;
        quant = io_buffer + scale_bytes;
        if(!read_at(fd, scales, scale_bytes, entry->a + first_group * 2) ||
           !read_at(fd, quant, quant_bytes, entry->b + flat / 2)) return -1;
#endif

        for(int local = 0; local < rows; local++) {
            int row = begin + local;
            int col = 0;
            float total = 0.0f;
            while(col < HIDDEN) {
                int element = row * HIDDEN + col;
                int group = element >> 6;
                int take = 64 - (element & 63);
                int32_t subtotal = 0;
                if(take > HIDDEN - col) take = HIDDEN - col;
#ifdef CASIOLLM_NANOLM_HYPER
#ifdef CASIOLLM_NANOLM_CG4
                {
                    uint8_t const *group_data = io_buffer +
                        (group - first_group) * CG4_GROUP_BYTES;
                    if(group - first_group + 1 < groups)
                        prefetch_cg4(group_data + CG4_GROUP_BYTES);
                    subtotal = dot_cg4_i16_exact(group_data + 4,
                        element & 63, state.xq + col, take, unpacked);
                    total += (float)subtotal * be_f32(group_data);
                }
#else
                subtotal = dot_q4_i16_exact(quant, flat, element,
                    state.xq + col, take, unpacked);
#endif
#else
                for(int j = 0; j < take; j++) {
                    int current = element + j;
                    uint8_t packed = quant[(current >> 1) - flat / 2];
                    int q = ((current & 1) ? (packed >> 4) : (packed & 15)) - 8;
                    subtotal += q * (int32_t)state.xq[col + j];
                }
#endif
#ifndef CASIOLLM_NANOLM_CG4
                total += (float)subtotal *
                    f16(le16(scales + (group - first_group) * 2));
#endif
                col += take;
            }
            if(total > state.choose_best) {
                state.choose_best = total;
                state.choose_best_id = row;
            }
        }
        state.choose_at += rows;
    }

    if(state.choose_at >= 20000) {
        state.choosing = 0;
        return 1;
    }
    return 0;
}

static void os_decode_to_state(int token_id)
{
    uint8_t entry[8];
    uint8_t piece[48];
    uint16_t length;
    int fd;
    int at = 0;
    state.decoded[0] = '\0';
    if(token_id < 0 || token_id >= ID_IM_START) return;
    fd = BFile_Open(u"\\\\fls0\\NANOLM.TOK", BFile_ReadOnly);
    if(fd < 0) return;
    if(BFile_Read(fd, entry, sizeof(entry), 16 + token_id * 8) != (int)sizeof(entry)) {
        BFile_Close(fd); return;
    }
    length = le16(entry + 4);
    if(length >= sizeof(piece)) length = sizeof(piece) - 1;
    if(BFile_Read(fd, piece, length, 160016 + (int)le32(entry)) != length) {
        BFile_Close(fd); return;
    }
    BFile_Close(fd);
    for(uint16_t i = 0; i < length && at + 1 < (int)sizeof(state.decoded);) {
        if(i + 3 <= length && piece[i] == 0xe2 && piece[i + 1] == 0x96 && piece[i + 2] == 0x81) {
            state.decoded[at++] = ' '; i += 3;
        }
        else if(i + 6 == length && memcmp(piece + i, "<0x", 3) == 0 && piece[i + 5] == '>') {
            int hi = piece[i + 3], lo = piece[i + 4];
            int value = (hi >= 'A' ? hi - 'A' + 10 : hi - '0') * 16 + (lo >= 'A' ? lo - 'A' + 10 : lo - '0');
            if(value >= 32 && value < 127) state.decoded[at++] = (char)value;
            i += 6;
        }
        else {
            unsigned char c = piece[i++];
            if(c >= 32 && c < 127) state.decoded[at++] = (char)c;
        }
    }
    state.decoded[at] = '\0';
}

static int __attribute__((unused)) os_step(int ignored)
{
    int fd;
    int result;
    (void)ignored;
    fd = BFile_Open(u"\\\\fls0\\NANOLM.Q4", BFile_ReadOnly);
    if(fd < 0) { set_error("Missing NANOLM.Q4"); return -2; }

    if(state.forward_active) {
        int kind = state.forward_kind;
        int token = state.forward_token;
        result = forward_microstep(fd);
        BFile_Close(fd);
        if(result < 0) { set_error("Q4 read or compute error"); return -2; }
        if(result == 0) return 0;
        if(kind == FORWARD_EMIT) {
            state.pending = -1;
            state.have_pending = 0;
            os_decode_to_state(token);
            return token + 1;
        }
        return 0;
    }

    if(state.choosing) {
        result = choose_token_microstep(fd);
        BFile_Close(fd);
        if(result < 0) { set_error("Q4 vocabulary read error"); return -2; }
        if(result == 0) return 0;
        state.pending = state.choose_best_id;
        if(state.pending == ID_IM_END || state.pending == ID_EOT) return -1;
        state.have_pending = 1;
        state.generated++;
        return 0;
    }

    if(state.prompt_at < state.prompt_len) {
        int token = state.prompt[state.prompt_at];
        int ok = forward_begin(fd, token, FORWARD_PROMPT);
        BFile_Close(fd);
        if(!ok) { set_error("Q4 read or context error"); return -2; }
        state.prompt_at++;
        return 0;
    }
    if(state.have_pending) {
        int emitted = state.pending;
        int ok = forward_begin(fd, emitted, FORWARD_EMIT);
        BFile_Close(fd);
        if(!ok) { set_error("Q4 read or context error"); return -2; }
        return 0;
    }
    if(state.generated >= MAX_RESPONSE_TOKENS) {
        BFile_Close(fd);
        return -1;
    }
    choose_token_begin();
    BFile_Close(fd);
    return 0;
}

#ifdef CASIOLLM_NANOLM_EXTREME
/* One old state-machine unit, with NANOLM.Q4 already open. Keeping this
   separate preserves the exact v0.3 legacy scheduler above. */
static int os_step_extreme_unit(int fd)
{
    int result;
    state.profile_units++;

#if defined(CASIOLLM_NANOLM_BATCH_PREFILL) && defined(CASIOLLM_NANOLM_CG4)
    if(prefill.active) {
        result = prefill_microstep(fd);
        if(result < 0) {
            set_error("CG4 batched prefill error");
            return -2;
        }
        return 0;
    }
#endif

    if(state.forward_active) {
        result = forward_microstep(fd);
        if(result < 0) { set_error("Q4 read or compute error"); return -2; }
        return 0;
    }

    if(state.choosing) {
        result = choose_token_microstep(fd);
        if(result < 0) { set_error("Q4 vocabulary read error"); return -2; }
        if(result == 0) return 0;
        state.pending = state.choose_best_id;
        if(state.pending == ID_IM_END || state.pending == ID_EOT) return -1;
        state.have_pending = 1;
        state.generated++;
        /* The token is already known. Display it now; its forward pass is
           only needed to calculate the following token. */
        os_decode_to_state(state.pending);
        return state.pending + 1;
    }

    if(state.prompt_at < state.prompt_len) {
#if defined(CASIOLLM_NANOLM_BATCH_PREFILL) && defined(CASIOLLM_NANOLM_CG4)
        if(!prefill_begin(fd)) {
            if(!assets.error[0]) set_error("CG4 prefill start error");
            return -2;
        }
#else
        int token = state.prompt[state.prompt_at++];
        if(!forward_begin(fd, token, FORWARD_PROMPT)) {
            set_error("Q4 read or context error");
            return -2;
        }
#endif
        return 0;
    }

    if(state.have_pending) {
        int emitted = state.pending;
        if(state.generated >= MAX_RESPONSE_TOKENS || state.position >= MAX_CONTEXT)
            return -1;
        state.have_pending = 0;
        if(!forward_begin(fd, emitted, FORWARD_EMIT)) {
            set_error("Q4 read or context error");
            return -2;
        }
        return 0;
    }

    if(state.generated >= MAX_RESPONSE_TOKENS) return -1;
    choose_token_begin();
    return 0;
}

/* Coalesce bounded microsteps under one OS visit. Ultra builds keep the model
   descriptor open across visits, eliminating thousands of Fugue opens. */
static int os_step_extreme(int ignored)
{
    int fd;
    int limit = CASIOLLM_COALESCE_STEPS;
    int result = 0;
    (void)ignored;
#ifdef CASIOLLM_NANOLM_CG4
    fd = assets.model_fd;
    if(fd < 0) { set_error("NANOLM.CG4 is not open"); return -2; }
#else
    fd = BFile_Open(u"\\\\fls0\\NANOLM.Q4", BFile_ReadOnly);
    if(fd < 0) { set_error("Missing NANOLM.Q4"); return -2; }
#endif
#ifdef CASIOLLM_NANOLM_BATCH_PREFILL
    if(prefill.active || state.prompt_at < state.prompt_len)
        limit = CASIOLLM_PREFILL_COALESCE_STEPS;
#endif
    for(int i = 0; i < limit && result == 0; i++)
        result = os_step_extreme_unit(fd);
#ifndef CASIOLLM_NANOLM_CG4
    BFile_Close(fd);
#endif
    return result;
}
#endif

int nanolm_step(int *token_id)
{
    int result;
    char line[192];
    if(!state.active) return -1;
#ifdef CASIOLLM_NANOLM_SHORT_IDENTITY
    if(state.canned_active) {
        int out = 0;
        if(!state.canned_reply[state.canned_at]) {
            uint32_t total = elapsed_ticks(state.started_ticks, rtc_ticks());
            state.active = 0;
            state.canned_active = 0;
            snprintf(line, sizeof(line),
                "END canned_identity chunks=%d first_ms=%lu total_ms=%lu",
                state.generated,
                (unsigned long)(state.first_token_ticks * 1000U / 128U),
                (unsigned long)(total * 1000U / 128U));
            log_event(line);
            return -1;
        }
        if(state.canned_reply[state.canned_at] == ' ')
            state.decoded[out++] = state.canned_reply[state.canned_at++];
        while(state.canned_reply[state.canned_at] &&
              state.canned_reply[state.canned_at] != ' ' &&
              out + 1 < (int)sizeof(state.decoded))
            state.decoded[out++] = state.canned_reply[state.canned_at++];
        state.decoded[out] = '\0';
        state.generated++;
        *token_id = -1;
        if(!state.first_token_recorded) {
            state.first_token_ticks = elapsed_ticks(state.started_ticks, rtc_ticks());
            state.first_token_recorded = 1;
        }
        return 1;
    }
#endif
#ifdef CASIOLLM_NANOLM_EXTREME
    state.profile_switches++;
    result = gint_world_switch(GINT_CALL(os_step_extreme, 0));
#else
    state.profile_switches++;
    result = gint_world_switch(GINT_CALL(os_step, 0));
#endif
    if(result > 0) {
        *token_id = result - 1;
        if(!state.first_token_recorded) {
            state.first_token_ticks = elapsed_ticks(state.started_ticks, rtc_ticks());
            state.first_token_recorded = 1;
        }
#ifdef CASIOLLM_NANOLM_HYPER
        if(state.generated > 0 && state.generated <= MAX_RESPONSE_TOKENS)
            state.token_trace[state.generated - 1] = *token_id;
#endif
        return 1;
    }
    if(result < 0) {
        state.active = 0;
        if(result == -2 && !state.error_logged) {
            snprintf(line, sizeof(line),
                "ERROR %s prompt=%d/%d pos=%d layer=%d stage=%d",
                assets.error, state.prompt_at, state.prompt_len, state.position,
                state.forward_layer, state.forward_stage);
            state.error_logged = 1;
            log_event(line);
            log_token_trace();
        }
        else if(result == -1) {
            uint32_t total = elapsed_ticks(state.started_ticks, rtc_ticks());
            snprintf(line, sizeof(line),
                "END backend=%s gen=%d pos=%d first_ms=%lu total_ms=%lu sw=%lu u=%lu",
                NANOLM_BACKEND_LOG, state.generated, state.position,
                (unsigned long)(state.first_token_ticks * 1000U / 128U),
                (unsigned long)(total * 1000U / 128U),
                (unsigned long)state.profile_switches,
                (unsigned long)state.profile_units);
            log_event(line);
            log_token_trace();
        }
        return -1;
    }
    return 0;
}

void nanolm_cancel(void)
{
    char line[192];
    uint32_t total = elapsed_ticks(state.started_ticks, rtc_ticks());
    state.active = 0;
    state.have_pending = 0;
    state.forward_active = 0;
    state.choosing = 0;
    snprintf(line, sizeof(line),
        "CANCEL backend=%s p=%d/%d pos=%d l=%d s=%d gen=%d first_ms=%lu total_ms=%lu sw=%lu u=%lu",
        NANOLM_BACKEND_LOG, state.prompt_at, state.prompt_len, state.position, state.forward_layer,
        state.forward_stage, state.generated,
        (unsigned long)(state.first_token_ticks * 1000U / 128U),
        (unsigned long)(total * 1000U / 128U),
        (unsigned long)state.profile_switches,
        (unsigned long)state.profile_units);
    log_event(line);
    log_token_trace();
}

void nanolm_decode_token(int token_id, char *out, size_t out_size)
{
    (void)token_id;
    if(!out_size) return;
    strncpy(out, state.decoded, out_size - 1);
    out[out_size - 1] = '\0';
}

char const *nanolm_error(void) { return assets.error; }

void nanolm_shutdown(void)
{
    if(state.active) nanolm_cancel();
    if(assets.model_fd >= 0)
        (void)gint_world_switch(GINT_CALL(os_close_model, 0));
    free(io_buffer);
    io_buffer = NULL;
    memset(&state, 0, sizeof(state));
    memset(&assets, 0, sizeof(assets));
}
