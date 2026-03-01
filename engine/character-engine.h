#pragma once

/**
 * Character Engine - Personality and behavior control via logit manipulation
 *
 * Controls model behavior at the tensor/logit level without relying on system
 * prompts or chat templates. Works with any model architecture.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct character_engine character_engine_t;

// Mood states that affect generation parameters
typedef enum {
    CHAR_MOOD_NEUTRAL    = 0,
    CHAR_MOOD_HAPPY      = 1,
    CHAR_MOOD_SAD        = 2,
    CHAR_MOOD_EXCITED    = 3,
    CHAR_MOOD_CALM       = 4,
    CHAR_MOOD_ANGRY      = 5,
    CHAR_MOOD_CURIOUS    = 6,
    CHAR_MOOD_CREATIVE   = 7,
    CHAR_MOOD_FOCUSED    = 8,
    CHAR_MOOD_CUSTOM     = 9,
} char_mood;

// Character personality definition
typedef struct {
    const char * name;
    const char * persona;          // persona description for context
    float        temperature;      // base temperature (0.1 - 2.0)
    float        top_p;            // base top-p
    float        repetition_penalty;
    float        creativity;       // 0.0 = conservative, 1.0 = very creative
    float        verbosity;        // 0.0 = terse, 1.0 = verbose
    float        formality;        // 0.0 = casual, 1.0 = formal
} char_personality;

// Logit bias entry - boost or suppress specific token patterns
typedef struct {
    int32_t  token_id;
    float    bias;     // positive = boost, negative = suppress
} char_logit_bias;

// Create / destroy
character_engine_t * character_engine_create(void);
void                 character_engine_free(character_engine_t * ce);

// Set personality
void                 character_engine_set_personality(character_engine_t * ce,
                                                       const char_personality * personality);

// Set mood (adjusts temperature, penalties, etc.)
void                 character_engine_set_mood(character_engine_t * ce, char_mood mood);

// Set custom mood parameters
void                 character_engine_set_custom_mood(character_engine_t * ce,
                                                       float temperature_mod,   // added to base temp
                                                       float top_p_mod,         // added to base top_p
                                                       float rep_penalty_mod);  // added to base penalty

// Logit bias management
void                 character_engine_add_logit_bias(character_engine_t * ce,
                                                      int32_t token_id, float bias);
void                 character_engine_clear_logit_biases(character_engine_t * ce);

// Token suppression (prevent specific tokens from being generated)
void                 character_engine_suppress_token(character_engine_t * ce, int32_t token_id);
void                 character_engine_clear_suppressions(character_engine_t * ce);

// Uncensored mode - suppresses refusal patterns at the logit level.
// When enabled with a vocab, scans for refusal tokens ("I cannot", "I'm sorry",
// "as an AI", etc.) and applies strong negative logit biases to prevent refusal.
// This works at the weight/sampling level, not just prompting.
void                 character_engine_set_uncensored(character_engine_t * ce, bool enabled);
bool                 character_engine_get_uncensored(const character_engine_t * ce);

// Supply refusal token IDs to suppress when uncensored mode is active.
// Call this after loading a model to scan the vocab for refusal patterns.
void                 character_engine_set_refusal_tokens(character_engine_t * ce,
                                                          const int32_t * token_ids,
                                                          int32_t n_tokens);

// Get effective sampling parameters (personality + mood + biases combined)
typedef struct {
    float    temperature;
    float    top_p;
    float    min_p;
    float    repetition_penalty;
    int32_t  top_k;
    int32_t  n_logit_biases;
    const char_logit_bias * logit_biases;
    int32_t  n_suppressed;
    const int32_t * suppressed_tokens;
    bool     uncensored;
} char_effective_params;

char_effective_params character_engine_get_params(const character_engine_t * ce);

// Get personality context string to prepend to prompts
char *               character_engine_get_context(const character_engine_t * ce);

// Free strings
void                 character_engine_free_string(char * str);

#ifdef __cplusplus
}
#endif
