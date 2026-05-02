// Unit tests for common/universal-toolcall.{h,cpp}.
//
// Runs canned tool-call payloads from every known GGUF format through the
// universal parser and asserts the extracted name+arguments. Compile for
// host (fast) and for Android arm64-v8a (adb push to /data/local/tmp).

#include "chat-parser.h"
#include "universal-toolcall.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct expected_tool_call {
    std::string name;
    json        arguments;
};

struct test_case {
    std::string                       label;
    std::string                       input;
    std::vector<expected_tool_call>   expected;
};

static int failures = 0;

static void run_case(const test_case & tc) {
    common_chat_parser_params syntax;
    syntax.format            = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    syntax.parse_tool_calls  = true;

    common_chat_msg_parser builder(tc.input, /* is_partial */ false, syntax);
    bool parsed = common_universal_toolcall_parse(builder);

    const auto & msg = builder.result();
    bool ok = msg.tool_calls.size() == tc.expected.size()
           && parsed == !tc.expected.empty();

    if (ok) {
        for (size_t i = 0; i < tc.expected.size(); ++i) {
            if (msg.tool_calls[i].name != tc.expected[i].name) { ok = false; break; }
            json actual_args;
            try { actual_args = json::parse(msg.tool_calls[i].arguments); }
            catch (...) { ok = false; break; }
            if (actual_args != tc.expected[i].arguments) { ok = false; break; }
        }
    }

    if (ok) {
        std::printf("  PASS  %s\n", tc.label.c_str());
    } else {
        ++failures;
        std::printf("  FAIL  %s\n", tc.label.c_str());
        std::printf("        parsed=%d  got %zu tool calls (expected %zu)\n",
                    (int)parsed, msg.tool_calls.size(), tc.expected.size());
        for (size_t i = 0; i < msg.tool_calls.size(); ++i) {
            std::printf("        [%zu] name=\"%s\" args=%s\n",
                        i,
                        msg.tool_calls[i].name.c_str(),
                        msg.tool_calls[i].arguments.c_str());
        }
    }
}

int main() {
    std::vector<test_case> cases = {
        {
            "qwen3-coder: web_search {query=news}",
            "<tool_call>\n<function=web_search>\n<parameter=query>latest ai news</parameter>\n</function>\n</tool_call>",
            { { "web_search", json{{"query", "latest ai news"}} } },
        },
        {
            "hermes-2-pro: <tool_call>{JSON}</tool_call>",
            "<tool_call>{\"name\": \"web_search\", \"arguments\": {\"query\": \"weather\"}}</tool_call>",
            { { "web_search", json{{"query", "weather"}} } },
        },
        {
            "hermes-2-pro: bare JSON with name",
            "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}",
            { { "get_weather", json{{"city", "Paris"}} } },
        },
        {
            "mistral: [TOOL_CALLS][{...}]",
            "[TOOL_CALLS][{\"name\": \"search\", \"arguments\": {\"q\": \"abc\"}}]",
            { { "search", json{{"q", "abc"}} } },
        },
        {
            "lfm2: <|tool_call_start|>[{...}]<|tool_call_end|>",
            "<|tool_call_start|>[{\"name\": \"get_time\", \"arguments\": {\"tz\": \"UTC\"}}]<|tool_call_end|>",
            { { "get_time", json{{"tz", "UTC"}} } },
        },
        {
            "minimax-m2",
            "<minimax:tool_call><invoke name=\"search\"><parameter name=\"q\">hello</parameter></invoke></minimax:tool_call>",
            { { "search", json{{"q", "hello"}} } },
        },
        {
            "seed-oss",
            "<seed:tool_call><function=search><parameter=q>hello</parameter></function></seed:tool_call>",
            { { "search", json{{"q", "hello"}} } },
        },
        {
            "qwen3-coder with multiple params",
            "<tool_call>\n<function=make_request>\n<parameter=url>https://x.com</parameter>\n<parameter=method>POST</parameter>\n</function>\n</tool_call>",
            { { "make_request", json{{"url", "https://x.com"}, {"method", "POST"}} } },
        },
        {
            "no tool call — plain text",
            "Hello world, this is just content.",
            {},
        },
        {
            "qwen3-coder with prose before the call",
            "Sure, I'll look that up for you.\n<tool_call>\n<function=web_search>\n<parameter=query>ToolNeuron</parameter>\n</function>\n</tool_call>",
            { { "web_search", json{{"query", "ToolNeuron"}} } },
        },
        {
            "hermes: ```json fenced tool call",
            "```json\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"abc\"}}\n```",
            { { "web_search", json{{"query", "abc"}} } },
        },
        {
            "hermes: <function=NAME>{args}</function>",
            "<function=calculator>{\"a\": 1, \"b\": 2}</function>",
            { { "calculator", json{{"a", 1}, {"b", 2}} } },
        },
        {
            "mistral: two tool calls",
            "[TOOL_CALLS][{\"name\":\"a\",\"arguments\":{}},{\"name\":\"b\",\"arguments\":{\"x\":1}}]",
            {
                { "a", json::object() },
                { "b", json{{"x", 1}} },
            },
        },
        {
            "no tool call — JSON without name field",
            "Here is a response: {\"result\": 42, \"status\": \"ok\"}",
            {},
        },
        {
            "no tool call — lone XML-ish tag",
            "Check <tool_call> or <function=> syntax.",
            {},
        },
        {
            "qwen3-coder: integer-looking arg kept as string",
            "<tool_call>\n<function=set_timer>\n<parameter=seconds>30</parameter>\n</function>\n</tool_call>",
            { { "set_timer", json{{"seconds", "30"}} } },
        },
        {
            "qwen3-coder: reasoning tag then tool call",
            "<think>user wants weather</think>\n<tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>",
            { { "get_weather", json{{"city", "Paris"}} } },
        },
        {
            "qwen3-coder: two sequential tool calls",
            "<tool_call>\n<function=a>\n<parameter=x>1</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n<parameter=y>2</parameter>\n</function>\n</tool_call>",
            {
                { "a", json{{"x", "1"}} },
                { "b", json{{"y", "2"}} },
            },
        },
        {
            "hermes: tool call with nested object arg",
            "<tool_call>{\"name\":\"http\",\"arguments\":{\"config\":{\"url\":\"x\",\"method\":\"GET\"}}}</tool_call>",
            { { "http", json{{"config", json{{"url","x"},{"method","GET"}}}} } },
        },
        {
            "degenerate: tool_call tag with mismatched content (user prose)",
            "I don't know. <tool_call>",
            {},
        },
    };

    std::printf("Running %zu universal-toolcall cases...\n", cases.size());
    for (const auto & tc : cases) run_case(tc);

    std::printf("\n%s: %d failures out of %zu cases\n",
                failures == 0 ? "OK" : "FAIL",
                failures, cases.size());

    return failures == 0 ? 0 : 1;
}
