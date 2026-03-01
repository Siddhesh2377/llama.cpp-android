#include "tool-manager.h"

#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>

struct tool_entry {
    std::string name;
    std::string description;
    std::vector<tool_param_def> params;
};

struct tool_manager {
    std::vector<tool_entry>  tools;
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
    tm->tools.push_back(std::move(entry));
}

void tool_manager_clear(tool_manager_t * tm) {
    if (tm) tm->tools.clear();
}

char * tool_manager_get_prompt(const tool_manager_t * tm) {
    if (!tm || tm->tools.empty()) return strdup_alloc("");

    std::string prompt;
    prompt += "You have access to the following tools. To use a tool, respond with a JSON object in this exact format:\n";
    prompt += "```json\n{\"tool\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n```\n\n";
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
    // skip whitespace and colon
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':' || json[pos] == '\t' || json[pos] == '\n')) pos++;

    if (pos >= json.size()) return "";

    if (json[pos] == '"') {
        // string value
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
    }
    return "";
}

// Attempt to find a JSON tool call in the model output
static bool try_parse_json_tool_call(const std::string & output,
                                      std::string & tool_name,
                                      std::string & args_json) {
    // Find any JSON object that contains "tool" or "name" + "arguments"
    for (size_t i = 0; i < output.size(); i++) {
        if (output[i] != '{') continue;
        size_t end = find_matching_brace(output, i);
        if (end == std::string::npos) continue;

        std::string obj = output.substr(i, end - i + 1);

        // try "tool" key first
        std::string name = extract_json_value(obj, "tool");
        if (name.empty()) name = extract_json_value(obj, "name");
        if (name.empty()) continue;

        std::string args = extract_json_value(obj, "arguments");
        if (args.empty()) args = extract_json_value(obj, "params");
        if (args.empty()) args = "{}";

        tool_name = name;
        args_json = args;
        return true;
    }
    return false;
}

// Attempt to find an XML-style tool call: <tool_call>...</tool_call>
static bool try_parse_xml_tool_call(const std::string & output,
                                     std::string & tool_name,
                                     std::string & args_json) {
    size_t start = output.find("<tool_call>");
    size_t end = output.find("</tool_call>");
    if (start == std::string::npos || end == std::string::npos || end <= start) return false;

    std::string content = output.substr(start + 11, end - start - 11);
    return try_parse_json_tool_call(content, tool_name, args_json);
}

// Attempt function-call style: function_name(args)
static bool try_parse_function_call(const std::string & output,
                                     const std::vector<tool_entry> & tools,
                                     std::string & tool_name,
                                     std::string & args_json) {
    for (const auto & tool : tools) {
        size_t pos = output.find(tool.name);
        if (pos == std::string::npos) continue;

        size_t paren = pos + tool.name.size();
        // skip whitespace
        while (paren < output.size() && output[paren] == ' ') paren++;
        if (paren >= output.size() || output[paren] != '(') continue;

        size_t close = output.find(')', paren);
        if (close == std::string::npos) continue;

        tool_name = tool.name;
        std::string raw_args = output.substr(paren + 1, close - paren - 1);
        if (!raw_args.empty() && raw_args[0] == '{') {
            args_json = raw_args;
        } else {
            args_json = "{\"input\": \"" + raw_args + "\"}";
        }
        return true;
    }
    return false;
}

tool_call_result tool_manager_parse_output(const tool_manager_t * tm, const char * model_output) {
    tool_call_result result = {};
    result.is_valid = false;

    if (!tm || !model_output) return result;

    std::string output(model_output);
    std::string tool_name, args_json;

    // Try strategies in order: JSON > XML > function-call
    if (try_parse_json_tool_call(output, tool_name, args_json) ||
        try_parse_xml_tool_call(output, tool_name, args_json) ||
        try_parse_function_call(output, tm->tools, tool_name, args_json)) {

        // verify tool exists
        bool found = false;
        for (const auto & t : tm->tools) {
            if (t.name == tool_name) {
                found = true;
                break;
            }
        }

        if (found) {
            result.tool_name = strdup_alloc(tool_name);
            result.arguments_json = strdup_alloc(args_json);
            result.is_valid = true;
        }
    }

    return result;
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
