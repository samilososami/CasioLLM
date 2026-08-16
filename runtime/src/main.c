/*
   LLM, a small local-chat UI for the Casio fx-CG50.

   Model weights are streamed from the calculator's storage. Every request
   starts a fresh model state. Generation is cooperative, so the next prompt
   can be edited while the current answer streams and F6 can stop it.
*/
#include <gint/clock.h>
#include <gint/display.h>
#include <gint/drivers/keydev.h>
#include <gint/keyboard.h>
#include <gint/rtc.h>

#include "nanolm.h"
#include "tinytalk.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define SCREEN_W 384
#define SCREEN_H 216
#define HEADER_H 20
#define INPUT_TOP 184
#define INPUT_H 32
#define CHAT_TOP 23
#define CHAT_BOTTOM (INPUT_TOP - 3)
#define LINE_H 10
#define MESSAGE_GAP 2
#define TURN_LINE_TOP_GAP 3
#define TURN_LINE_BOTTOM_GAP 3
#define WRAP_CHARS 44
#define INPUT_VISIBLE_CHARS 39
#define MAX_MESSAGES 40
#define MAX_MESSAGE_LEN 384
#define MAX_INPUT_LEN 80

#define COLOR_HEADER C_RGB(4, 8, 17)
#define COLOR_PANEL C_RGB(28, 29, 31)
#define COLOR_BORDER C_RGB(10, 12, 15)
/* Closest RGB565 values to #002A70 and #468DF2. */
#define COLOR_USER C_RGB(0, 10, 14)
#define COLOR_MODEL C_RGB(9, 35, 29)
#define COLOR_MUTED C_RGB(12, 12, 12)
#define COLOR_SEND C_RGB(1, 25, 29)
#define COLOR_SEND_BORDER C_RGB(0, 16, 24)
#define COLOR_SEND_DISABLED C_RGB(9, 18, 9)
#define COLOR_SEND_DISABLED_BORDER C_RGB(14, 28, 14)
#define COLOR_SEND_DISABLED_PLANE C_RGB(22, 44, 22)

typedef struct {
    bool is_user;
    bool turn_complete;
    int duration_seconds;
    char text[MAX_MESSAGE_LEN];
} message_t;

static message_t messages[MAX_MESSAGES];
static int message_count;
static char input[MAX_INPUT_LEN + 1];
static int input_len;
static int cursor;
/* Number of pixels held above the bottom of the transcript. */
static int scroll_px;
static keydev_t *keyboard;
static bool caps_lock;
static bool alpha_pending;
static bool shift_pending;
static int held_shortcut;
static bool generation_active;
static bool generation_answer_started;
static bool display_dirty = true;
static int generation_thinking_phase = -1;
static uint32_t generation_started_ticks;
static bool generation_timer_valid;

static bool model_start(char const *prompt)
{
#ifdef CASIOLLM_BACKEND_NANOLM
    return nanolm_start(prompt);
#else
    return tinytalk_start(prompt);
#endif
}

static int model_step(int *token_id)
{
#ifdef CASIOLLM_BACKEND_NANOLM
    return nanolm_step(token_id);
#else
    return tinytalk_step(token_id);
#endif
}

static void model_cancel(void)
{
#ifdef CASIOLLM_BACKEND_NANOLM
    nanolm_cancel();
#else
    tinytalk_cancel();
#endif
}

static void model_decode_token(int token_id, char *out, size_t out_size)
{
#ifdef CASIOLLM_BACKEND_NANOLM
    nanolm_decode_token(token_id, out, out_size);
#else
    tinytalk_decode_token(token_id, out, out_size);
#endif
}

static char const *model_error(void)
{
#ifdef CASIOLLM_BACKEND_NANOLM
    return nanolm_error();
#else
    return tinytalk_error();
#endif
}

static void model_shutdown(void)
{
#ifdef CASIOLLM_BACKEND_NANOLM
    nanolm_shutdown();
#else
    tinytalk_shutdown();
#endif
}

static unsigned long generation_elapsed_seconds(void)
{
    if(!generation_timer_valid) return 0;
    return (unsigned long)((uint32_t)(rtc_ticks() - generation_started_ticks) / 128U);
}

static int text_line_count(char const *text)
{
    int len = (int)strlen(text);
    int lines = (len + WRAP_CHARS - 1) / WRAP_CHARS;
    return lines ? lines : 1;
}

static int message_line_height(int index)
{
    return messages[index].is_user ? LINE_H + 1 : LINE_H;
}

static int gap_before_message(int index)
{
    if(index == 0) return 0;
    if(messages[index - 1].is_user) return MESSAGE_GAP;
    return TURN_LINE_BOTTOM_GAP;
}

static void copy_visual_line(char *out, char const *text, int line)
{
    int start = line * WRAP_CHARS;
    int remaining = (int)strlen(text) - start;
    int length = remaining > WRAP_CHARS ? WRAP_CHARS : remaining;

    if(length < 0) length = 0;
    memcpy(out, text + start, length);
    out[length] = '\0';
}

static int chat_height(bool thinking)
{
    int height = 0;

    for(int i = 0; i < message_count; i++) {
        height += gap_before_message(i);
        height += text_line_count(messages[i].text) * message_line_height(i);
        if(!messages[i].is_user && messages[i].turn_complete)
            height += TURN_LINE_TOP_GAP + 1;
    }
    if(thinking) {
        if(height) height += MESSAGE_GAP;
        height += LINE_H;
    }
    return height;
}

static void draw_chat(bool thinking, bool generating)
{
    int total = chat_height(thinking);
    int visible = CHAT_BOTTOM - CHAT_TOP;
    int first = total - visible - scroll_px;
    int position = 0;
    char line[WRAP_CHARS + 1];

    if(first < 0) first = 0;
    if(scroll_px > total - visible && total > visible)
        scroll_px = total - visible;
    if(scroll_px < 0) scroll_px = 0;

    for(int i = 0; i < message_count; i++) {
        int lines = text_line_count(messages[i].text);
        int line_height = message_line_height(i);
        int color = messages[i].is_user ? COLOR_USER : COLOR_MODEL;
        position += gap_before_message(i);
        for(int j = 0; j < lines; j++) {
            int line_y = position + j * line_height;
            if(line_y + line_height <= first || line_y >= first + visible) continue;
            copy_visual_line(line, messages[i].text, j);
            dtext(5, CHAT_TOP + line_y - first, color, line);
            /* A one-pixel horizontal overprint gives user prompts a restrained
               semibold footprint without introducing a large UI font. */
            if(messages[i].is_user)
                dtext(6, CHAT_TOP + line_y - first, color, line);
        }
        position += lines * line_height;
        if(!messages[i].is_user && messages[i].duration_seconds >= 0) {
            char marker[24];
            int width;
            int last_line_y = position - line_height;
            copy_visual_line(line, messages[i].text, lines - 1);
            dsize(line, NULL, &width, NULL);
            snprintf(marker, sizeof(marker), "[%ds]",
                messages[i].duration_seconds);
            if(last_line_y + line_height > first && last_line_y < first + visible)
                dtext(5 + width + 6, CHAT_TOP + last_line_y - first,
                    COLOR_MUTED, marker);
        }
        if(!messages[i].is_user && messages[i].turn_complete) {
            int separator_y = position + TURN_LINE_TOP_GAP;
            if(separator_y >= first && separator_y < first + visible)
                dline(5, CHAT_TOP + separator_y - first,
                    SCREEN_W - 6, CHAT_TOP + separator_y - first, C_BLACK);
            position += TURN_LINE_TOP_GAP + 1;
        }
    }
    if(generating && message_count > 0 &&
       !messages[message_count - 1].is_user) {
        int width;
        int last_line_y = position - LINE_H;
        int lines = text_line_count(messages[message_count - 1].text);
        copy_visual_line(line, messages[message_count - 1].text, lines - 1);
        dsize(line, NULL, &width, NULL);
        if(last_line_y + LINE_H > first && last_line_y < first + visible)
            dcircle(5 + width + 6, CHAT_TOP + last_line_y - first + 4,
                3, COLOR_MODEL, COLOR_MODEL);
    }
    if(thinking) {
        static char const *dots[] = { ".", "..", "..." };
        char label[40];
        int phase = (int)((rtc_ticks() / 64) % 3);
        snprintf(label, sizeof(label), "thinking%s %lus", dots[phase],
            generation_elapsed_seconds());
        if(position) position += MESSAGE_GAP;
        if(position + LINE_H > first && position < first + visible)
            dtext(5, CHAT_TOP + position - first, COLOR_MUTED, label);
    }

    if(first > 0) dtext(SCREEN_W - 35, CHAT_TOP, COLOR_MUTED, "^ more");
    if(first + visible < total) dtext(SCREEN_W - 35, CHAT_BOTTOM - LINE_H, COLOR_MUTED, "v more");
}

static void draw_input(void)
{
    int cursor_x;
    int start = cursor > INPUT_VISIBLE_CHARS ? cursor - INPUT_VISIBLE_CHARS : 0;
    int displayed = input_len - start;
    char before[MAX_INPUT_LEN + 1];
    char visible[INPUT_VISIBLE_CHARS + 1];

    if(displayed > INPUT_VISIBLE_CHARS) displayed = INPUT_VISIBLE_CHARS;
    memcpy(visible, input + start, displayed);
    visible[displayed] = '\0';

    drect(0, INPUT_TOP, SCREEN_W - 1, SCREEN_H - 1, COLOR_PANEL);
    drect_border(0, INPUT_TOP, SCREEN_W - 1, SCREEN_H - 1,
        COLOR_PANEL, 1, COLOR_BORDER);
    dtext(5, INPUT_TOP + 10, C_BLACK, "> ");
    dtext(20, INPUT_TOP + 9, C_BLACK, visible);
    memcpy(before, input + start, cursor - start);
    before[cursor - start] = '\0';
    dsize(before, NULL, &cursor_x, NULL);
    drect(20 + cursor_x, INPUT_TOP + 20, 22 + cursor_x, INPUT_TOP + 22,
        C_BLACK);

    {
        int background = generation_active ? COLOR_SEND_DISABLED : COLOR_SEND;
        int border = generation_active ? COLOR_SEND_DISABLED_BORDER : COLOR_SEND_BORDER;
        int plane = generation_active ? COLOR_SEND_DISABLED_PLANE : C_WHITE;
        /* Two filled wings reproduce the supplied concave paper-plane
           silhouette: broad tail, center notch, and long right-hand tip. */
        int upper_x[] = { 353, 375, 363 };
        int upper_y[] = { 192, 200, 200 };
        int lower_x[] = { 363, 375, 353 };
        int lower_y[] = { 200, 200, 208 };

        drect_border(348, 188, 379, 212, background, 1, border);
        /* Clip four pixels to soften the button's rectangular corners. */
        dpixel(348, 188, COLOR_PANEL);
        dpixel(379, 188, COLOR_PANEL);
        dpixel(348, 212, COLOR_PANEL);
        dpixel(379, 212, COLOR_PANEL);
        dpoly(upper_x, upper_y, 3, plane, C_NONE);
        dpoly(lower_x, lower_y, 3, plane, C_NONE);
    }
}

static void render(void)
{
    bool thinking = generation_active && !generation_answer_started;
    bool generating = generation_active && generation_answer_started;
    dclear(C_WHITE);
    drect(0, 0, SCREEN_W - 1, HEADER_H - 1, COLOR_HEADER);
#ifdef CASIOLLM_BACKEND_NANOLM
    dtext(6, 5, C_WHITE, "CasioLLM  |  NanoLM extreme");
#else
    dtext(6, 5, C_WHITE, "CasioLLM  |  TinyTalk v1 extreme");
#endif
    if(caps_lock) {
        drect(331, 4, 345, 15, C_RED);
        dtext(335, 5, C_WHITE, "A");
        dtext(351, 5, C_WHITE, "LOCK");
    }
    draw_chat(thinking, generating);
    draw_input();
    dupdate();
}

static void add_message(bool is_user, char const *text)
{
    if(message_count == MAX_MESSAGES) {
        memmove(messages, messages + 1, sizeof(messages[0]) * (MAX_MESSAGES - 1));
        message_count--;
    }
    messages[message_count].is_user = is_user;
    messages[message_count].turn_complete = false;
    messages[message_count].duration_seconds = -1;
    strncpy(messages[message_count].text, text, MAX_MESSAGE_LEN - 1);
    messages[message_count].text[MAX_MESSAGE_LEN - 1] = '\0';
    message_count++;
    scroll_px = 0;
}

static void append_to_last_message(char const *text)
{
    size_t have;
    size_t room;
    if(message_count == 0 || !text) return;
    have = strlen(messages[message_count - 1].text);
    room = MAX_MESSAGE_LEN - 1 - have;
    if(room) strncat(messages[message_count - 1].text, text, room);
}

static void append_generation_duration(void)
{
    if(!generation_timer_valid) return;
    if(message_count == 0 || messages[message_count - 1].is_user)
        add_message(false, "");
    messages[message_count - 1].duration_seconds =
        (int)generation_elapsed_seconds();
    messages[message_count - 1].turn_complete = true;
    generation_timer_valid = false;
}

static void insert_char(char c)
{
    if(input_len >= MAX_INPUT_LEN) return;
    memmove(input + cursor + 1, input + cursor, input_len - cursor + 1);
    input[cursor++] = c;
    input_len++;
}

static void erase_before_cursor(void)
{
    if(cursor == 0) return;
    memmove(input + cursor - 1, input + cursor, input_len - cursor + 1);
    cursor--;
    input_len--;
}

static void move_word_left(void)
{
    while(cursor > 0 && input[cursor - 1] == ' ') cursor--;
    while(cursor > 0 && input[cursor - 1] != ' ') cursor--;
}

static void move_word_right(void)
{
    while(cursor < input_len && input[cursor] == ' ') cursor++;
    while(cursor < input_len && input[cursor] != ' ') cursor++;
}

/* CASIO's printed ALPHA character layout. */
static char alpha_char(int key)
{
    switch(key) {
        case KEY_XOT: return 'a'; case KEY_LOG: return 'b';
        case KEY_LN: return 'c'; case KEY_SIN: return 'd';
        case KEY_COS: return 'e'; case KEY_TAN: return 'f';
        case KEY_FRAC: return 'g'; case KEY_FD: return 'h';
        case KEY_LEFTP: return 'i'; case KEY_RIGHTP: return 'j';
        case KEY_COMMA: return 'k'; case KEY_ARROW: return 'l';
        case KEY_7: return 'm'; case KEY_8: return 'n';
        case KEY_9: return 'o'; case KEY_4: return 'p';
        case KEY_5: return 'q'; case KEY_6: return 'r';
        case KEY_MUL: return 's'; case KEY_DIV: return 't';
        case KEY_1: return 'u'; case KEY_2: return 'v';
        case KEY_3: return 'w'; case KEY_ADD: return 'x';
        case KEY_SUB: return 'y'; case KEY_0: return 'z';
        case KEY_DOT: return ' ';
        default: return '\0';
    }
}

static char direct_char(int key, bool alpha)
{
    int digit = keycode_digit(key);
    if(alpha) {
        char c = alpha_char(key);
        if(c) return c;
    }
    if(digit >= 0) return '0' + digit;

    switch(key) {
        case KEY_DOT: return shift_pending ? '?' : '.';
        case KEY_COMMA: return shift_pending ? ':' : ',';
        case KEY_LEFTP: return '(';
        case KEY_RIGHTP: return ')';
        case KEY_ADD: return '+';
        case KEY_SUB: return '-';
        case KEY_MUL: return '*';
        case KEY_DIV: return '/';
        default: return '\0';
    }
}

static void stop_generation(void)
{
    if(!generation_active) return;
    model_cancel();
    if(!generation_answer_started) add_message(false, "[stopped]");
    append_generation_duration();
    generation_active = false;
    generation_answer_started = false;
    generation_thinking_phase = -1;
    display_dirty = true;
}

static void start_prompt(void)
{
    /* EXE must not change the meaning of a prompt merely because the cursor
       left one or more spaces at the end. NanoLM is unusually sensitive to
       that distinction ("hi" and "hi " tokenize differently), so normalize
       the submitted text before it is displayed, logged or tokenized. */
    while(input_len > 0 && input[input_len - 1] == ' ') input_len--;
    input[input_len] = '\0';
    if(cursor > input_len) cursor = input_len;
    if(input_len == 0 || generation_active) return;
    generation_started_ticks = rtc_ticks();
    generation_timer_valid = true;
    add_message(true, input);
    input[0] = '\0';
    input_len = 0;
    cursor = 0;

    /* Paint the submitted prompt and timer before the first coalesced model
       block. Otherwise EXE appears unresponsive for the duration of a block. */
    generation_active = true;
    generation_answer_started = false;
    generation_thinking_phase = -1;
    display_dirty = true;
    render();
    display_dirty = false;

    if(!model_start(messages[message_count - 1].text)) {
        add_message(false, model_error()[0] ? model_error() : "Model could not start.");
        append_generation_duration();
        generation_active = false;
        display_dirty = true;
        return;
    }
    display_dirty = true;
}

static void advance_generation(void)
{
    char piece[32];
    int token;
    int status;
    if(!generation_active) return;
    status = model_step(&token);
    if(status < 0) {
        if(!generation_answer_started && model_error()[0])
            add_message(false, model_error());
        else if(!generation_answer_started)
            add_message(false, "[no output]");
        append_generation_duration();
        generation_active = false;
        generation_answer_started = false;
        generation_thinking_phase = -1;
        display_dirty = true;
        return;
    }
    if(status == 0) return;
    model_decode_token(token, piece, sizeof(piece));
    if(!piece[0]) return;
    if(!generation_answer_started) {
        add_message(false, "");
        generation_answer_started = true;
    }
    append_to_last_message(piece);
    display_dirty = true;
}

static bool handle_key(key_event_t event)
{
    char c;
    bool alpha;
    bool word_shortcut;

    if(event.type == KEYEV_NONE) return false;
    if(event.key == KEY_F6 && generation_active &&
       (event.type == KEYEV_DOWN || event.type == KEYEV_HOLD)) {
        stop_generation();
        return false;
    }
    if(event.key == KEY_SHIFT) {
        if(event.type == KEYEV_DOWN) shift_pending = true;
        return false;
    }
    if(event.key == KEY_ALPHA) {
        if(event.type == KEYEV_DOWN) {
            if(shift_pending) {
                caps_lock = !caps_lock;
                alpha_pending = false;
                shift_pending = false;
            }
            else alpha_pending = true;
            display_dirty = true;
        }
        return false;
    }

    word_shortcut = event.type == KEYEV_HOLD
        ? held_shortcut == event.key
        : shift_pending && (event.key == KEY_DEL || event.key == KEY_ACON);
    if(word_shortcut && event.key == KEY_DEL) {
        move_word_left();
        if(event.type == KEYEV_DOWN) held_shortcut = KEY_DEL;
        shift_pending = false;
        alpha_pending = false;
        display_dirty = true;
        return false;
    }
    if(word_shortcut && event.key == KEY_ACON) {
        move_word_right();
        if(event.type == KEYEV_DOWN) held_shortcut = KEY_ACON;
        shift_pending = false;
        alpha_pending = false;
        display_dirty = true;
        return false;
    }
    if(event.type == KEYEV_DOWN) held_shortcut = 0;

    if(event.key == KEY_EXIT) {
        if(generation_active) model_cancel();
        return true;
    }
    /* EXE is intentionally ignored while a response is being generated. The
       text stays in the input field and can be sent as soon as generation
       ends or is stopped with F6. */
    if(event.key == KEY_EXE) {
        if(!generation_active) start_prompt();
        return false;
    }
    if(event.key == KEY_LEFT) {
        if(cursor > 0) cursor--;
        display_dirty = true;
        return false;
    }
    if(event.key == KEY_RIGHT) {
        if(cursor < input_len) cursor++;
        display_dirty = true;
        return false;
    }
    if(event.key == KEY_UP) {
        scroll_px += LINE_H;
        display_dirty = true;
        return false;
    }
    if(event.key == KEY_DOWN) {
        if(scroll_px > 0) scroll_px -= LINE_H;
        display_dirty = true;
        return false;
    }
    if(event.key == KEY_DEL) {
        erase_before_cursor();
        shift_pending = false;
        display_dirty = true;
        return false;
    }
    if(event.key == KEY_ACON) {
        input[0] = '\0';
        input_len = cursor = 0;
        display_dirty = true;
        return false;
    }
    alpha = caps_lock || alpha_pending;
    c = direct_char(event.key, alpha);
    if(c) {
        insert_char(c);
        alpha_pending = false;
        shift_pending = false;
        display_dirty = true;
        return false;
    }

    alpha_pending = false;
    shift_pending = false;
    return false;
}

int main(void)
{
    keydev_transform_t previous_transform;
    keydev_transform_t input_transform;

    keyboard = keydev_std();
    previous_transform = keydev_transform(keyboard);
    input_transform = previous_transform;
    /* Keep raw SHIFT/ALPHA events so their native key sequence remains
       observable across the whole input loop, while enabling all-key repeat. */
    input_transform.enabled &= ~(KEYDEV_TR_DELAYED_MODS
        | KEYDEV_TR_INSTANT_MODS | KEYDEV_TR_DELETE_MODIFIERS);
    input_transform.enabled |= KEYDEV_TR_REPEATS | KEYDEV_TR_DELETE_RELEASES;
    keydev_set_transform(keyboard, input_transform);
    keydev_set_standard_repeats(keyboard, 500000, 70000);

    while(true) {
        key_event_t event;
        if(generation_active && !generation_answer_started) {
            int phase = (int)((rtc_ticks() / 64) % 3);
            if(phase != generation_thinking_phase) {
                generation_thinking_phase = phase;
                display_dirty = true;
            }
        }
        if(display_dirty) {
            render();
            display_dirty = false;
        }
        if(generation_active) {
            /* Model reads temporarily switch away from gint. Leave one full
               128 Hz keyboard-scan interval between inference microsteps so
               ordinary taps, not only long-held F6, enter the event queue. */
#ifdef CASIOLLM_NANOLM_EXTREME
            /* One inference call now coalesces several old microsteps. One
               keyboard scan here therefore replaces several 9 ms windows. */
            sleep_us(9000);
#else
            sleep_us(9000);
#endif
            if(keydev_keydown(keyboard, KEY_F6)) {
                stop_generation();
                continue;
            }
        }
        event = keydev_read(keyboard, !generation_active, NULL);
        if(handle_key(event)) break;
        if(generation_active) advance_generation();
    }
    model_shutdown();
    keydev_set_transform(keyboard, previous_transform);
    return 1;
}
