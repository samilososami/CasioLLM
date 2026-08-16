/* Host adapter for executing src/nanolm.c against the real calculator
   assets. It is intentionally small: the model runtime itself remains the
   source under test. */
#include "nanolm.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static char const *q4_path;
static char const *idx_path;
static char const *tok_path;
static char const *tri_path;
static char const *pfx_path;
static char const *log_path;
static FILE *handles[12];
static uint64_t started_ns;

static int path_has(uint16_t const *path, char const *needle)
{
    char ascii[96];
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
    if(path_has(path, "NANOLM.Q4")) return q4_path;
    if(path_has(path, "NANOLM.IDX")) return idx_path;
    if(path_has(path, "NANOLM.TOK")) return tok_path;
    if(path_has(path, "NANOLM.TRI")) return tri_path;
    if(path_has(path, "NANOLM.PFX")) return pfx_path;
    if(path_has(path, "NANOLM.LOG")) return log_path;
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

uint32_t rtc_ticks(void)
{
    struct timespec now;
    uint64_t elapsed;
    clock_gettime(CLOCK_MONOTONIC, &now);
    elapsed = (uint64_t)now.tv_sec * 1000000000ULL + (uint64_t)now.tv_nsec;
    if(!started_ns) started_ns = elapsed;
    return (uint32_t)((elapsed - started_ns) * 128ULL / 1000000000ULL);
}

int main(int argc, char **argv)
{
    char piece[64];
    int cycles = getenv("CASIOLLM_REOPEN_TEST") ? 3 : 1;
    int failed = 0;
    int token;
    int status;
    int stop_after = 0;
    if(argc != 8) {
        fprintf(stderr,
            "usage: %s NANOLM.Q4 NANOLM.IDX NANOLM.TOK NANOLM.TRI NANOLM.PFX NANOLM.LOG prompt\n",
            argv[0]);
        return 2;
    }
    q4_path = argv[1];
    idx_path = argv[2];
    tok_path = argv[3];
    tri_path = argv[4];
    pfx_path = argv[5];
    log_path = argv[6];
#ifdef CASIOLLM_HOST_TOOLS
    if(strcmp(argv[7], "--build-prefix") == 0) {
        if(!nanolm_debug_build_prefix()) {
            fprintf(stderr, "prefix build failed: %s\n", nanolm_error());
            return 1;
        }
        fprintf(stderr, "prefix cache written to %s\n", pfx_path);
        nanolm_shutdown();
        return 0;
    }
#endif
    if(getenv("CASIOLLM_STOP_AFTER_FIRST")) stop_after = 1;
    for(int cycle = 0; cycle < cycles; cycle++) {
        int steps = 0;
        int tokens = 0;
        if(!nanolm_start(argv[7])) {
            fprintf(stderr, "cycle=%d start failed: %s\n",
                cycle + 1, nanolm_error());
            nanolm_shutdown();
            return 1;
        }
        while(steps++ < 100000) {
            status = nanolm_step(&token);
            if(status < 0) break;
            if(status == 0) continue;
            nanolm_decode_token(token, piece, sizeof(piece));
            fputs(piece, stdout);
            fflush(stdout);
            tokens++;
            if(stop_after && tokens >= stop_after) {
                nanolm_cancel();
                break;
            }
        }
        putchar('\n');
        fprintf(stderr, "cycle=%d steps=%d tokens=%d error=%s\n",
            cycle + 1, steps, tokens, nanolm_error());
        if(steps >= 100000 || nanolm_error()[0]) failed = 1;
        nanolm_shutdown();
        if(failed) break;
    }
    return failed;
}
