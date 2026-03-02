#include "tool-manager.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <cstdlib>

struct tool_entry {
    std::string name;
    std::string description;
    std::vector<tool_param_def> params;
};

struct tool_manager {
    std::vector<tool_entry>  tools;
    // O(1) tool lookup by name (index into tools vector)
    std::unordered_map<std::string, size_t> tool_index;
    tool_execute_callback    callback  = nullptr;
    void                   * user_data = nullptr;
};

static char * strdup_alloc(const std::string & s) {
    char * p = (char *)malloc(s.size() + 1);
    if (p) memcpy(p, s.c_str(), s.size() + 1);
    return p;
}

tool_manager_t * tool_manager_create(void) {
    return new tool_manager();
}

void tool_manager_free(tool_manager_t * tm) {
    delete tm;
}

void tool_manager_register(tool_manager_t * tm, const tool_def * tool) {
    if (!tm || !tool) return;

    tool_entry entry;
    entry.name = tool->name ? tool->name : "";
    entry.description = tool->description ? tool->description : "";
    for (int i = 0; i < tool->n_params; i++) {
        entry.params.push_back(tool->params[i]);
    }
    size_t idx = tm->tools.size();
    tm->tools.push_back(std::move(entry));
    // build O(1) index for lookup during parse (use tools[idx] since entry was moved)
    if (!tm->tools[idx].name.empty()) {
        tm->tool_index[tm->tools[idx].name] = idx;
    }
}

void tool_manager_clear(tool_manager_t * tm) {
    if (!tm) return;
    tm->tools.clear();
    tm->tool_index.clear();
}

char * tool_manager_get_prompt(const tool_manager_t * tm) {
    if (!tm || tm->tools.empty()) return strdup_alloc("");

    std::string prompt;
    prompt += "You have access to the following tools. To use a tool, respond with a JSON object in this exact format:\n";
    prompt += "```json\n{\"tool\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n```\n\n";
    prompt += "You may call multiple tools in a single response by using multiple JSON objects.\n\n";
    prompt += "Available tools:\n\n";

    for (const auto & tool : tm->tools) {
        prompt += "### " + tool.name + "\n";
        prompt += tool.description + "\n";
        if (!tool.params.empty()) {
            prompt += "Parameters:\n";
            for (const auto & p : tool.params) {
                const char * type_str = "string";
                switch (p.type) {
                    case TOOL_PARAM_NUMBER:  type_str = "number"; break;
                    case TOOL_PARAM_BOOLEAN: type_str = "boolean"; break;
                    case TOOL_PARAM_ARRAY:   type_str = "array"; break;
                    case TOOL_PARAM_OBJECT:  type_str = "object"; break;
                    default: break;
                }
                prompt += "- `" + std::string(p.name) + "` (" + type_str + ")";
                if (p.required) prompt += " [required]";
                prompt += ": " + std::string(p.description ? p.description : "") + "\n";
            }
        }
        prompt += "\n";
    }

    prompt += "If no tool is needed, respond normally without the JSON format.\n";

    return strdup_alloc(prompt);
}

// Simple JSON string extraction (avoids regex for portability)
static size_t find_matching_brace(const std::string & s, size_t start) {
    if (start >= s.size() || s[start] != '{') return std::string::npos;
    int depth = 0;
    bool in_string = false;
    for (size_t i = start; i < s.size(); i++) {
        char c = s[i];
        if (in_string) {
            if (c == '\\') { i++; continue; }
            if (c == '"') in_string = false;
        } else {
            if (c == '"') in_string = true;
            else if (c == '{') depth++;
            else if (c == '}') { depth--; if (depth == 0) return i; }
        }
    }
    return std::string::npos;
}

static std::string extract_json_value(const std::string & json, const std::string & key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos += search.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':' || json[pos] == '\t' || json[pos] == '\n')) pos++;

    if (pos >= json.size()) return "";

    if (json[pos] == '"') {
        size_t end = pos + 1;
        while (end < json.size()) {
            if (json[end] == '\\') { end += 2; continue; }
            if (json[end] == '"') break;
            end++;
        }
        return json.substr(pos + 1, end - pos - 1);
    } else if (json[pos] == '{') {
        size_t end = find_matching_brace(json, pos);
        if (end != std::string::npos) {
            return json.substr(pos, end - pos + 1);
        }
    } else if (json[pos] == '[') {
        // handle array values
        int depth = 0;
        for (size_t i = pos; i < json.size(); i++) {
            if (json[i] == '[') depth++;
            else if (json[i] == ']') { depth--; if (depth == 0) return json.substr(pos, i - pos + 1); }
        }
    }
    return "";
}

// Validate required parameters are present in args_json
static bool validate_params(const tool_entry & tool, const std::string & args_json) {
    for (const auto & p : tool.params) {
        if (!p.required) continue;
        if (!p.name) continue;
        // check if the parameter name appears as a key in the JSON
        std::string key = "\"" + std::string(p.name) + "\"";
        if (args_json.find(key) == std::string::npos) {
            return false;
        }
    }
    return true;
}

// try to parse a single JSON tool call from a JSON object string
static bool parse_single_json_call(const std::string & obj,
                                    std::string & tool_name,
                                    std::string & args_json) {
    std::string name = extract_json_value(obj, "tool");
    if (name.empty()) name = extract_json_value(obj, "name");
    if (name.empty()) name = extract_json_value(obj, "function");
    if (name.empty()) return false;

    std::string args = extract_json_value(obj, "arguments");
    if (args.empty()) args = extract_json_value(obj, "params");
    if (args.empty()) args = extract_json_value(obj, "parameters");
    if (args.empty()) args = "{}";

    tool_name = name;
    args_json = args;
    return true;
}

// find ALL JSON tool calls in the output (supports multiple tool calls per response)
static std::vector<std::pair<std::string, std::string>> find_all_json_tool_calls(
    const std::string & output) {

    std::vector<std::pair<std::string, std::string>> calls;
    size_t pos = 0;
    while (pos < output.size()) {
        size_t brace = output.find('{', pos);
        if (brace == std::string::npos) break;

        size_t end = find_matching_brace(output, brace);
        if (end == std::string::npos) { pos = brace + 1; continue; }

        std::string obj = output.substr(brace, end - brace + 1);
        std::string name, args;
        if (parse_single_json_call(obj, name, args)) {
            calls.push_back({name, args});
        }
        pos = end + 1;
    }
    return calls;
}

// find all XML-style tool calls: <tool_call>...</tool_call>
static std::vector<std::pair<std::string, std::string>> find_all_xml_tool_calls(
    const std::string & output) {

    std::vector<std::pair<std::string, std::string>> calls;
    size_t pos = 0;
    while (pos < output.size()) {
        size_t start = output.find("<tool_call>", pos);
        if (start == std::string::npos) break;
        size_t end = output.find("</tool_call>", start);
        if (end == std::string::npos) break;

        std::string content = output.substr(start + 11, end - start - 11);
        std::string name, args;
        if (parse_single_json_call(content, name, args)) {
            calls.push_back({name, args});
        }
        pos = end + 12;
    }
    return calls;
}

// find function-call style: function_name(args) — only for registered tool names
static std::vector<std::pair<std::string, std::string>> find_all_function_calls(
    const std::string & output,
    const std::vector<tool_entry> & tools) {

    std::vector<std::pair<std::string, std::string>> calls;
    for (const auto & tool : tools) {
        size_t pos = 0;
        while (pos < output.size()) {
            size_t found = output.find(tool.name, pos);
            if (found == std::string::npos) break;

            size_t paren = found + tool.name.size();
            while (paren < output.size() && output[paren] == ' ') paren++;
            if (paren >= output.size() || output[paren] != '(') { pos = found + 1; continue; }

            size_t close = output.find(')', paren);
            if (close == std::string::npos) break;

            std::string raw_args = output.substr(paren + 1, close - paren - 1);
            std::string args_json;
            if (!raw_args.empty() && raw_args[0] == '{') {
                args_json = raw_args;
            } else {
                args_json = "{\"input\": \"" + raw_args + "\"}";
            }
            calls.push_back({tool.name, args_json});
            pos = close + 1;
        }
    }
    return calls;
}

// parse output for the first valid tool call (backward compatible)
tool_call_result tool_manager_parse_output(const tool_manager_t * tm, const char * model_output) {
    tool_call_result result = {};
    result.is_valid = false;

    if (!tm || !model_output) return result;

    std::string output(model_output);

    // try strategies in order: JSON > XML > function-call
    auto calls = find_all_json_tool_calls(output);
    if (calls.empty()) calls = find_all_xml_tool_calls(output);
    if (calls.empty()) calls = find_all_function_calls(output, tm->tools);

    for (auto & [name, args] : calls) {
        // O(1) tool existence check via hash map
        auto it = tm->tool_index.find(name);
        if (it == tm->tool_index.end()) continue;

        const tool_entry & tool = tm->tools[it->second];
        // validate required params are present
        if (!validate_params(tool, args)) continue;

        result.tool_name = strdup_alloc(name);
        result.arguments_json = strdup_alloc(args);
        result.is_valid = true;
        return result;
    }

    return result;
}

// parse output for ALL valid tool calls (multiple tools per response)
tool_call_result * tool_manager_parse_output_all(const tool_manager_t * tm,
                                                  const char * model_output,
                                                  int32_t * n_calls) {
    if (n_calls) *n_calls = 0;
    if (!tm || !model_output || !n_calls) return nullptr;

    std::string output(model_output);

    auto calls = find_all_json_tool_calls(output);
    if (calls.empty()) calls = find_all_xml_tool_calls(output);
    if (calls.empty()) calls = find_all_function_calls(output, tm->tools);

    // filter to valid, registered tools with correct params
    std::vector<std::pair<std::string, std::string>> valid;
    for (auto & [name, args] : calls) {
        auto it = tm->tool_index.find(name);
        if (it == tm->tool_index.end()) continue;
        if (!validate_params(tm->tools[it->second], args)) continue;
        valid.push_back({name, args});
    }

    if (valid.empty()) return nullptr;

    auto * results = (tool_call_result *)malloc(valid.size() * sizeof(tool_call_result));
    if (!results) return nullptr;

    for (size_t i = 0; i < valid.size(); i++) {
        results[i].tool_name = strdup_alloc(valid[i].first);
        results[i].arguments_json = strdup_alloc(valid[i].second);
        results[i].is_valid = true;
    }
    *n_calls = (int32_t)valid.size();
    return results;
}

void tool_manager_free_results(tool_call_result * results, int32_t n_calls) {
    if (!results) return;
    for (int32_t i = 0; i < n_calls; i++) {
        free((void *)results[i].tool_name);
        free((void *)results[i].arguments_json);
    }
    free(results);
}

void tool_manager_set_callback(tool_manager_t * tm, tool_execute_callback cb, void * user_data) {
    if (!tm) return;
    tm->callback = cb;
    tm->user_data = user_data;
}

char * tool_manager_execute(tool_manager_t * tm, const tool_call_result * call) {
    if (!tm || !call || !call->is_valid || !tm->callback) {
        return strdup_alloc("{\"error\": \"invalid call or no callback\"}");
    }
    const char * result = tm->callback(call->tool_name, call->arguments_json, tm->user_data);
    return result ? strdup_alloc(std::string(result)) : strdup_alloc("");
}

void tool_manager_free_string(char * str) {
    free(str);
}
