#ifndef CASIOLLM_TINYTALK_H
#define CASIOLLM_TINYTALK_H

#include <stdbool.h>
#include <stddef.h>

/* TinyTalk is prepared lazily, only after it is selected on the model screen. */
bool tinytalk_prepare(void);
bool tinytalk_start(char const *user_prompt);
int tinytalk_step(int *token_id);
void tinytalk_decode_token(int token_id, char *out, size_t out_size);
char const *tinytalk_error(void);
void tinytalk_cancel(void);
void tinytalk_shutdown(void);

#ifdef CASIOLLM_HOST_TOOLS
int tinytalk_debug_encode(char const *text, int *tokens, int capacity);
void tinytalk_debug_decode(int token_id, char *out, size_t out_size);
bool tinytalk_debug_build_prefix(void);
#endif

#endif
