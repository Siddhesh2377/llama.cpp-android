#include "ggml-engine.h"
#include "tool-manager.h"
#include "character-engine.h"

#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <cstdlib>
#include <string>

// ANSI colors for terminal output
#define CLR_RESET  "\033[0m"
#define CLR_GREEN  "\033[32m"
#define CLR_YELLOW "\033[33m"
#define CLR_RED    "\033[31m"
#define CLR_CYAN   "\033[36m"
#define CLR_BOLD   "\033[1m"

static void print_header(const char * title) {
    printf("\n%s%s=== %s ===%s\n\n", CLR_BOLD, CLR_CYAN, title, CLR_RESET);
}

static void print_pass(const char * test) {
    printf("  %s[PASS]%s %s\n", CLR_GREEN, CLR_RESET, test);
}

static void print_fail(const char * test, const char * reason) {
    printf("  %s[FAIL]%s %s - %s\n", CLR_RED, CLR_RESET, test, reason);
}

static void print_info(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    printf("  %s[INFO]%s ", CLR_YELLOW, CLR_RESET);
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
}

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, name, reason) do { \
    if (cond) { print_pass(name); tests_passed++; } \
    else { print_fail(name, reason); tests_failed++; } \
} while(0)

// Token callback for generation test
static bool token_callback(const char * text, void * user_data) {
    std::string * output = (std::string *)user_data;
    *output += text;
    printf("%s", text);
    fflush(stdout);
    return true;
}

// ---- Test: Engine lifecycle ----
static void test_engine_lifecycle() {
    print_header("Engine Lifecycle");

    auto params = ggml_engine_default_params();
    TEST_ASSERT(params.n_batch == 512, "default params batch", "expected 512");
    TEST_ASSERT(params.use_mmap == true, "default params mmap", "expected true");

    auto * engine = ggml_engine_create(params);
    TEST_ASSERT(engine != nullptr, "engine create", "returned null");
    TEST_ASSERT(!ggml_engine_is_loaded(engine), "engine not loaded initially", "should not be loaded");

    ggml_engine_free(engine);
    print_pass("engine free");
    tests_passed++;
}

// ---- Test: Model loading ----
static void test_model_loading(const char * model_path) {
    print_header("Model Loading");

    auto params = ggml_engine_default_params();
    params.n_ctx = 2048;
    params.n_threads = 4;

    auto * engine = ggml_engine_create(params);

    auto status = ggml_engine_load_model(engine, model_path);
    TEST_ASSERT(status == GGML_ENGINE_OK, "model load", "failed to load model");
    TEST_ASSERT(ggml_engine_is_loaded(engine), "model is loaded", "should be loaded after load");

    // test context info
    int32_t ctx_size = ggml_engine_context_size(engine);
    TEST_ASSERT(ctx_size > 0, "context size > 0", "expected positive context size");
    print_info("Context size: %d", ctx_size);

    int32_t ctx_used = ggml_engine_context_used(engine);
    TEST_ASSERT(ctx_used == 0, "context used == 0 initially", "expected 0");

    ggml_engine_free(engine);
}

// ---- Test: Model info JSON ----
static void test_model_info(const char * model_path) {
    print_header("Model Info JSON");

    auto params = ggml_engine_default_params();
    params.n_ctx = 512;
    auto * engine = ggml_engine_create(params);
    ggml_engine_load_model(engine, model_path);

    char * json = ggml_engine_model_info_json(engine);
    TEST_ASSERT(json != nullptr, "model info not null", "returned null");
    TEST_ASSERT(strlen(json) > 10, "model info has content", "JSON too short");

    // check it contains expected fields
    TEST_ASSERT(strstr(json, "n_embd") != nullptr, "has n_embd field", "missing n_embd");
    TEST_ASSERT(strstr(json, "n_layer") != nullptr, "has n_layer field", "missing n_layer");
    TEST_ASSERT(strstr(json, "n_vocab") != nullptr, "has n_vocab field", "missing n_vocab");
    TEST_ASSERT(strstr(json, "metadata") != nullptr, "has metadata field", "missing metadata");

    print_info("Model info JSON length: %zu bytes", strlen(json));
    // print first 500 chars
    printf("  %.500s%s\n", json, strlen(json) > 500 ? "..." : "");

    ggml_engine_free_string(json);
    ggml_engine_free(engine);
}

// ---- Test: Tokenization ----
static void test_tokenization(const char * model_path) {
    print_header("Tokenization");

    auto params = ggml_engine_default_params();
    params.n_ctx = 512;
    auto * engine = ggml_engine_create(params);
    ggml_engine_load_model(engine, model_path);

    const char * test_text = "Hello, world! This is a test of tokenization.";
    int32_t tokens[128];
    int32_t n = ggml_engine_tokenize(engine, test_text, tokens, 128);
    TEST_ASSERT(n > 0, "tokenize returns positive count", "expected > 0 tokens");
    print_info("Tokenized '%s' into %d tokens", test_text, n);

    // detokenize
    char * detok = ggml_engine_detokenize(engine, tokens, n);
    TEST_ASSERT(detok != nullptr, "detokenize not null", "returned null");
    TEST_ASSERT(strlen(detok) > 0, "detokenize has content", "empty result");
    print_info("Detokenized: '%s'", detok);

    ggml_engine_free_string(detok);
    ggml_engine_free(engine);
}

// ---- Test: Text generation ----
static void test_generation(const char * model_path) {
    print_header("Text Generation");

    auto params = ggml_engine_default_params();
    params.n_ctx = 2048;
    params.n_threads = 4;
    auto * engine = ggml_engine_create(params);
    ggml_engine_load_model(engine, model_path);

    auto sampling = ggml_engine_default_sampling();
    sampling.n_predict = 64;
    sampling.temperature = 0.7f;

    std::string output;
    printf("  Generating: ");
    auto status = ggml_engine_generate(engine, "Once upon a time", sampling, token_callback, &output);
    printf("\n");

    TEST_ASSERT(status == GGML_ENGINE_OK, "generation status OK", "generation failed");
    TEST_ASSERT(!output.empty(), "output not empty", "no output generated");
    print_info("Generated %zu chars", output.length());

    // check response getter
    char * resp = ggml_engine_get_response(engine);
    TEST_ASSERT(resp != nullptr, "get_response not null", "returned null");
    TEST_ASSERT(strlen(resp) > 0, "get_response has content", "empty");
    ggml_engine_free_string(resp);

    // perf
    auto perf = ggml_engine_get_perf(engine);
    TEST_ASSERT(perf.prompt_tokens > 0, "perf prompt tokens > 0", "expected > 0");
    TEST_ASSERT(perf.generated_tokens > 0, "perf generated tokens > 0", "expected > 0");
    print_info("Prompt: %d tokens, %.1f ms (%.1f t/s)",
        perf.prompt_tokens, perf.prompt_eval_ms, perf.prompt_tokens_per_sec);
    print_info("Generation: %d tokens, %.1f ms (%.1f t/s)",
        perf.generated_tokens, perf.generation_ms, perf.generation_tokens_per_sec);

    ggml_engine_free(engine);
}

// ---- Test: Tool Manager ----
static void test_tool_manager() {
    print_header("Tool Manager");

    auto * tm = tool_manager_create();
    TEST_ASSERT(tm != nullptr, "tool manager create", "returned null");

    // register a tool
    tool_param_def params[] = {
        { "city", "The city name", TOOL_PARAM_STRING, true },
    };
    tool_def weather_tool = {
        "get_weather",
        "Get the current weather for a city",
        params,
        1
    };
    tool_manager_register(tm, &weather_tool);
    print_pass("tool registered");
    tests_passed++;

    // get prompt
    char * prompt = tool_manager_get_prompt(tm);
    TEST_ASSERT(prompt != nullptr, "tool prompt not null", "returned null");
    TEST_ASSERT(strstr(prompt, "get_weather") != nullptr, "prompt contains tool name", "missing tool name");
    print_info("Tool prompt length: %zu", strlen(prompt));
    tool_manager_free_string(prompt);

    // test JSON parsing
    const char * model_output = "I'll check the weather. {\"tool\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}";
    auto result = tool_manager_parse_output(tm, model_output);
    TEST_ASSERT(result.is_valid, "JSON tool call parsed", "failed to parse");
    if (result.is_valid) {
        TEST_ASSERT(strcmp(result.tool_name, "get_weather") == 0, "correct tool name", "wrong tool name");
        TEST_ASSERT(strstr(result.arguments_json, "Tokyo") != nullptr, "correct args", "missing city");
        print_info("Parsed: tool=%s args=%s", result.tool_name, result.arguments_json);
        free((void*)result.tool_name);
        free((void*)result.arguments_json);
    }

    // test XML parsing
    const char * xml_output = "<tool_call>{\"tool\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>";
    auto xml_result = tool_manager_parse_output(tm, xml_output);
    TEST_ASSERT(xml_result.is_valid, "XML tool call parsed", "failed to parse XML");
    if (xml_result.is_valid) {
        print_info("XML parsed: tool=%s", xml_result.tool_name);
        free((void*)xml_result.tool_name);
        free((void*)xml_result.arguments_json);
    }

    tool_manager_free(tm);
}

// ---- Test: Character Engine ----
static void test_character_engine() {
    print_header("Character Engine");

    auto * ce = character_engine_create();
    TEST_ASSERT(ce != nullptr, "character engine create", "returned null");

    // set personality
    char_personality personality = {};
    personality.name = "Assistant";
    personality.persona = "You are a helpful and friendly AI assistant.";
    personality.temperature = 0.8f;
    personality.top_p = 0.9f;
    personality.repetition_penalty = 1.1f;
    personality.creativity = 0.6f;
    personality.verbosity = 0.5f;
    personality.formality = 0.3f;

    character_engine_set_personality(ce, &personality);
    print_pass("personality set");
    tests_passed++;

    // test mood effects
    character_engine_set_mood(ce, CHAR_MOOD_HAPPY);
    auto happy_params = character_engine_get_params(ce);
    TEST_ASSERT(happy_params.temperature > 0.8f, "happy mood increases temp", "expected higher temp");
    print_info("Happy mood temp: %.2f", happy_params.temperature);

    character_engine_set_mood(ce, CHAR_MOOD_FOCUSED);
    auto focused_params = character_engine_get_params(ce);
    TEST_ASSERT(focused_params.temperature < happy_params.temperature, "focused cooler than happy", "expected lower temp");
    print_info("Focused mood temp: %.2f", focused_params.temperature);

    // test logit biases
    character_engine_add_logit_bias(ce, 100, 5.0f);
    character_engine_add_logit_bias(ce, 200, -5.0f);
    auto biased_params = character_engine_get_params(ce);
    TEST_ASSERT(biased_params.n_logit_biases == 2, "2 logit biases", "expected 2");

    // test suppression
    character_engine_suppress_token(ce, 50256);
    auto supp_params = character_engine_get_params(ce);
    TEST_ASSERT(supp_params.n_suppressed == 1, "1 suppressed token", "expected 1");

    // test context generation
    char * ctx = character_engine_get_context(ce);
    TEST_ASSERT(ctx != nullptr, "context not null", "returned null");
    TEST_ASSERT(strlen(ctx) > 0, "context has content", "empty context");
    print_info("Context: %.200s", ctx);
    character_engine_free_string(ctx);

    character_engine_free(ce);
}

// ---- Test: Generation with Character Engine ----
static void test_character_generation(const char * model_path) {
    print_header("Character Engine + Generation");

    auto * ce = character_engine_create();
    char_personality personality = {};
    personality.name = "Sage";
    personality.persona = "You are Sage, a wise and knowledgeable philosopher.";
    personality.temperature = 0.9f;
    personality.top_p = 0.95f;
    personality.repetition_penalty = 1.15f;
    personality.creativity = 0.7f;
    personality.verbosity = 0.6f;
    personality.formality = 0.7f;

    character_engine_set_personality(ce, &personality);
    character_engine_set_mood(ce, CHAR_MOOD_CURIOUS);

    auto char_params = character_engine_get_params(ce);
    char * char_ctx = character_engine_get_context(ce);

    // build prompt with character context
    std::string prompt;
    if (char_ctx && strlen(char_ctx) > 0) {
        prompt += char_ctx;
        prompt += "\n\nUser: What is the meaning of life?\nSage:";
    } else {
        prompt = "User: What is the meaning of life?\nAssistant:";
    }
    character_engine_free_string(char_ctx);

    // engine
    auto params = ggml_engine_default_params();
    params.n_ctx = 2048;
    params.n_threads = 4;
    auto * engine = ggml_engine_create(params);
    ggml_engine_load_model(engine, model_path);

    auto sampling = ggml_engine_default_sampling();
    sampling.temperature = char_params.temperature;
    sampling.top_p = char_params.top_p;
    sampling.repeat_penalty = char_params.repetition_penalty;
    sampling.n_predict = 128;

    std::string output;
    printf("  Sage says: ");
    auto status = ggml_engine_generate(engine, prompt.c_str(), sampling, token_callback, &output);
    printf("\n");

    TEST_ASSERT(status == GGML_ENGINE_OK, "character generation OK", "generation failed");
    TEST_ASSERT(!output.empty(), "character output not empty", "no output");

    auto perf = ggml_engine_get_perf(engine);
    print_info("Generation: %d tokens at %.1f t/s", perf.generated_tokens, perf.generation_tokens_per_sec);

    character_engine_free(ce);
    ggml_engine_free(engine);
}

static void print_usage(const char * prog) {
    printf("Usage: %s -m <model_path> [options]\n", prog);
    printf("\nOptions:\n");
    printf("  -m <path>      Path to GGUF model file (required for model tests)\n");
    printf("  --all          Run all tests (default)\n");
    printf("  --no-model     Skip tests that require a model\n");
    printf("  --quick        Quick test (lifecycle + tool + character only)\n");
    printf("  -h, --help     Show this help\n");
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    bool run_model_tests = true;
    bool quick_mode = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--no-model") == 0) {
            run_model_tests = false;
        } else if (strcmp(argv[i], "--quick") == 0) {
            quick_mode = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    printf("%s%s\n", CLR_BOLD, "╔══════════════════════════════════════════╗");
    printf("║   Tool-Neuron Engine Test Suite          ║");
    printf("\n╚══════════════════════════════════════════╝%s\n", CLR_RESET);

    // Always run these
    test_engine_lifecycle();
    test_tool_manager();
    test_character_engine();

    if (run_model_tests && model_path) {
        if (!quick_mode) {
            test_model_loading(model_path);
            test_model_info(model_path);
            test_tokenization(model_path);
            test_generation(model_path);
            test_character_generation(model_path);
        } else {
            test_model_loading(model_path);
            test_model_info(model_path);
        }
    } else if (run_model_tests && !model_path) {
        print_info("Skipping model tests (no -m <path> provided)");
    }

    // Summary
    printf("\n%s%s══════════════════════════════════════════%s\n", CLR_BOLD, CLR_CYAN, CLR_RESET);
    printf("  Results: %s%d passed%s, %s%d failed%s\n",
        CLR_GREEN, tests_passed, CLR_RESET,
        tests_failed > 0 ? CLR_RED : CLR_GREEN, tests_failed, CLR_RESET);
    printf("%s%s══════════════════════════════════════════%s\n", CLR_BOLD, CLR_CYAN, CLR_RESET);

    return tests_failed > 0 ? 1 : 0;
}
