#ifndef CASIOLLM_NANOLM_H
#define CASIOLLM_NANOLM_H

#include <stdbool.h>
#include <stddef.h>

/* Loads the two small external assets. The large Q4 file remains on storage. */
bool nanolm_prepare(void);

/* Starts an entirely new request. No prior prompt or KV state is retained. */
bool nanolm_start(char const *user_prompt);

/* Advances at most one model token. Returns 1 for a token, 0 while preparing,
   and -1 when complete or when an error occurs. */
int nanolm_step(int *token_id);

/* Converts an output token into printable text for the CG50 transcript. */
void nanolm_decode_token(int token_id, char *out, size_t out_size);

char const *nanolm_error(void);
void nanolm_cancel(void);
void nanolm_shutdown(void);

#ifdef CASIOLLM_HOST_TOOLS
/* Build the exact fixed-prefix cache with this C runtime. The host BFile
   adapter maps NANOLM.PFX to the requested workspace file. */
bool nanolm_debug_build_prefix(void);
#endif

#endif
