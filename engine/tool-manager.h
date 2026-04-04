#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tool_manager tool_manager_t;

typedef enum {
    TOOL_PARAM_STRING,
    TOOL_PARAM_NUMBER,
    TOOL_PARAM_BOOLEAN,
    TOOL_PARAM_ARRAY,
    TOOL_PARAM_OBJECT,
} tool_param_type;

typedef struct {
    const char     * name;
    const char     * description;
    tool_param_type  type;
    bool             required;
} tool_param_def;

typedef struct {
    const char     * name;
    const char     * description;
    tool_param_def * params;
    int32_t          n_params;
} tool_def;

typedef struct {
    const char * tool_name;
    const char * arguments_json;
    bool         is_valid;
} tool_call_result;

// Return tool result string for the given call
typedef const char * (*tool_execute_callback)(const char * tool_name,
                                               const char * args_json,
                                               void * user_data);

tool_manager_t * tool_manager_create(void);
void             tool_manager_free(tool_manager_t * tm);

void             tool_manager_register(tool_manager_t * tm, const tool_def * tool);
void             tool_manager_clear(tool_manager_t * tm);

// Build tool description prompt for injection into system/user message
char *           tool_manager_get_prompt(const tool_manager_t * tm);

// Parse first tool call from model output (JSON, XML, function-call formats)
tool_call_result tool_manager_parse_output(const tool_manager_t * tm, const char * model_output);

// Parse all tool calls; caller frees with tool_manager_free_results
tool_call_result * tool_manager_parse_output_all(const tool_manager_t * tm,
                                                  const char * model_output,
                                                  int32_t * n_calls);

void tool_manager_free_results(tool_call_result * results, int32_t n_calls);

void             tool_manager_set_callback(tool_manager_t * tm,
                                            tool_execute_callback cb, void * user_data);

char *           tool_manager_execute(tool_manager_t * tm, const tool_call_result * call);

void             tool_manager_free_string(char * str);

#ifdef __cplusplus
}
#endif
