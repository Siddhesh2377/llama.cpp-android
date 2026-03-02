#pragma once

/**
 * ToolManager - Model-agnostic tool calling system
 *
 * Provides reliable tool calling that works across different models
 * without depending on specific chat templates or GBNF grammars.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tool_manager tool_manager_t;

// Tool parameter type
typedef enum {
    TOOL_PARAM_STRING,
    TOOL_PARAM_NUMBER,
    TOOL_PARAM_BOOLEAN,
    TOOL_PARAM_ARRAY,
    TOOL_PARAM_OBJECT,
} tool_param_type;

// Tool parameter definition
typedef struct {
    const char     * name;
    const char     * description;
    tool_param_type  type;
    bool             required;
} tool_param_def;

// Tool definition
typedef struct {
    const char     * name;
    const char     * description;
    tool_param_def * params;
    int32_t          n_params;
} tool_def;

// Tool call result from parsing model output
typedef struct {
    const char * tool_name;
    const char * arguments_json;  // JSON string of arguments
    bool         is_valid;
} tool_call_result;

// Callback when a tool is called: return the tool result as a string
typedef const char * (*tool_execute_callback)(const char * tool_name,
                                               const char * args_json,
                                               void * user_data);

// Create / destroy
tool_manager_t * tool_manager_create(void);
void             tool_manager_free(tool_manager_t * tm);

// Register tools
void             tool_manager_register(tool_manager_t * tm, const tool_def * tool);
void             tool_manager_clear(tool_manager_t * tm);

// Generate the tool description prompt to inject into the system/user message
char *           tool_manager_get_prompt(const tool_manager_t * tm);

// Parse model output for the first tool call (supports JSON, XML, and function-call formats)
tool_call_result tool_manager_parse_output(const tool_manager_t * tm, const char * model_output);

// Parse model output for ALL tool calls (multiple tools per response).
// Returns heap-allocated array of results. Caller must free with tool_manager_free_results.
tool_call_result * tool_manager_parse_output_all(const tool_manager_t * tm,
                                                  const char * model_output,
                                                  int32_t * n_calls);

// Free an array of tool_call_result returned by parse_output_all
void tool_manager_free_results(tool_call_result * results, int32_t n_calls);

// Set execution callback
void             tool_manager_set_callback(tool_manager_t * tm,
                                            tool_execute_callback cb, void * user_data);

// Execute a parsed tool call using the registered callback
char *           tool_manager_execute(tool_manager_t * tm, const tool_call_result * call);

// Free strings returned by tool_manager functions
void             tool_manager_free_string(char * str);

#ifdef __cplusplus
}
#endif
