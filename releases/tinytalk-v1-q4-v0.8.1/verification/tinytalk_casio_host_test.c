/* Runs src/tinytalk.c itself on the host with small BFile/gint adapters. */
#include "tinytalk.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static char const *model_path;
static char const *tokenizer_path;
static char const *log_path;
static char const *prefix_path;
static FILE *handles[8];
static uint64_t started_ns;

uint32_t rtc_ticks(void)
{
    struct timespec now;
    uint64_t elapsed;
    clock_gettime(CLOCK_MONOTONIC, &now);
    elapsed = (uint64_t)now.tv_sec * 1000000000ULL + (uint64_t)now.tv_nsec;
    if(!started_ns) started_ns = elapsed;
    return (uint32_t)((elapsed - started_ns) * 128ULL / 1000000000ULL);
}

static int path_has(uint16_t const *path, char const *needle)
{
    char ascii[80];
    int at = 0;
    while(path[at] && at + 1 < (int)sizeof(ascii)) {
        ascii[at] = (char)path[at];
        at++;
    }
    ascii[at] = '\0';
    return strstr(ascii, needle) != NULL;
}

static char const *resolve(uint16_t const *path)
{
    if(path_has(path, "TINYTLK.BIN")) return model_path;
    if(path_has(path, "TINYTLK.TOK")) return tokenizer_path;
    if(path_has(path, "TINYTLK.PFX")) return prefix_path;
    if(path_has(path, "CASIOLLM.LOG")) return log_path;
    return NULL;
}

int BFile_Open(uint16_t const *path, int mode)
{
    char const *real = resolve(path);
    char const *open_mode = mode ? "r+b" : "rb";
    if(!real) return -1;
    for(int i = 1; i < (int)(sizeof(handles) / sizeof(handles[0])); i++) {
        if(!handles[i]) {
            handles[i] = fopen(real, open_mode);
            return handles[i] ? i : -1;
        }
    }
    return -1;
}

int BFile_Close(int fd)
{
    int result;
    if(fd <= 0 || fd >= (int)(sizeof(handles) / sizeof(handles[0])) || !handles[fd])
        return -1;
    result = fclose(handles[fd]);
    handles[fd] = NULL;
    return result;
}

int BFile_Size(int fd)
{
    long current;
    long size;
    if(fd <= 0 || fd >= (int)(sizeof(handles) / sizeof(handles[0])) || !handles[fd])
        return -1;
    current = ftell(handles[fd]);
    fseek(handles[fd], 0, SEEK_END);
    size = ftell(handles[fd]);
    fseek(handles[fd], current, SEEK_SET);
    return (int)size;
}

int BFile_Read(int fd, void *buffer, int size, int offset)
{
    if(fd <= 0 || fd >= (int)(sizeof(handles) / sizeof(handles[0])) || !handles[fd])
        return -1;
    if(fseek(handles[fd], offset, SEEK_SET) != 0) return -1;
    return (int)fread(buffer, 1, (size_t)size, handles[fd]);
}

int BFile_Write(int fd, void const *buffer, int size)
{
    if(fd <= 0 || fd >= (int)(sizeof(handles) / sizeof(handles[0])) || !handles[fd])
        return -1;
    return (int)fwrite(buffer, 1, (size_t)size, handles[fd]);
}

int BFile_Seek(int fd, int offset)
{
    if(fd <= 0 || fd >= (int)(sizeof(handles) / sizeof(handles[0])) || !handles[fd])
        return -1;
    return fseek(handles[fd], offset, SEEK_SET) == 0 ? offset : -1;
}

int BFile_Create(uint16_t const *path, int type, int *size)
{
    char const *real = resolve(path);
    FILE *file;
    (void)type;
    (void)size;
    if(!real) return -1;
    file = fopen(real, "wb");
    if(!file) return -1;
    fclose(file);
    return 0;
}

int BFile_Remove(uint16_t const *path)
{
    char const *real = resolve(path);
    return real ? remove(real) : -1;
}

static int run_once(char const *prompt, int stop_after, int run)
{
    char piece[64];
    int steps = 0;
    int tokens = 0;
    int token;
    int status;
    int success;
    if(!tinytalk_start(prompt)) {
        fprintf(stderr, "run=%d start failed: %s\n", run, tinytalk_error());
        return 0;
    }
    while(steps++ < 100000) {
        status = tinytalk_step(&token);
        if(status < 0) break;
        if(status == 0) continue;
        tinytalk_decode_token(token, piece, sizeof(piece));
        fputs(piece, stdout);
        fflush(stdout);
        tokens++;
        if(stop_after && tokens >= stop_after) {
            tinytalk_cancel();
            break;
        }
    }
    putchar('\n');
    fprintf(stderr, "run=%d steps=%d tokens=%d error=%s\n",
        run, steps, tokens, tinytalk_error());
    success = steps < 100000 && !tinytalk_error()[0];
    tinytalk_shutdown();
    return success;
}

int main(int argc, char **argv)
{
    int stop_after = 0;
    int runs = 1;
    if(argc != 5 && argc != 6) {
        fprintf(stderr, "usage: %s model.bin tokenizer.bin log prompt [prefix.bin]\n", argv[0]);
        return 2;
    }
    model_path = argv[1];
    tokenizer_path = argv[2];
    log_path = argv[3];
    prefix_path = argc == 6 ? argv[5] : NULL;
#ifdef CASIOLLM_HOST_TOOLS
    if(getenv("CASIOLLM_BUILD_PREFIX")) {
        if(!prefix_path) {
            fprintf(stderr, "prefix output path is required\n");
            return 2;
        }
        if(!tinytalk_debug_build_prefix()) {
            fprintf(stderr, "prefix build failed: %s\n", tinytalk_error());
            tinytalk_shutdown();
            return 1;
        }
        fprintf(stderr, "prefix cache written to %s\n", prefix_path);
        tinytalk_shutdown();
        return 0;
    }
    if(getenv("CASIOLLM_DUMP_TOKENS")) {
        int tokens[128];
        int count = tinytalk_debug_encode(argv[4], tokens, 128);
        if(count < 0) {
            fprintf(stderr, "encode failed: %s\n", tinytalk_error());
            tinytalk_shutdown();
            return 1;
        }
        for(int i = 0; i < count; i++) {
            char piece[64];
            tinytalk_debug_decode(tokens[i], piece, sizeof(piece));
            printf("%d\t%d\t%s\n", i, tokens[i], piece);
        }
        tinytalk_shutdown();
        return 0;
    }
#endif
    if(getenv("CASIOLLM_STOP_AFTER_FIRST")) stop_after = 1;
    if(getenv("CASIOLLM_REOPEN_TEST")) runs = 3;
    for(int run = 1; run <= runs; run++) {
        if(!run_once(argv[4], stop_after, run)) return 1;
    }
    return 0;
}
