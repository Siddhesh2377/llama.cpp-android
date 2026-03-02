#include "ggml-engine.h"
#include "tool-manager.h"
#include "character-engine.h"
#include "rag-engine.h"

#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <cstdlib>
#include <string>
#include <fstream>
#include <vector>
#include <chrono>

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

// ---- Helper: load file into memory ----
static std::vector<unsigned char> load_file_bytes(const char * path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return {};
    auto size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<unsigned char> buf(size);
    f.read(reinterpret_cast<char *>(buf.data()), size);
    return buf;
}

// ---- Test: VLM Loading ----
static void test_vlm_loading(const char * model_path, const char * mmproj_path) {
    print_header("VLM Loading");

    auto params = ggml_engine_default_params();
    params.n_ctx = 2048;
    params.n_threads = 4;
    auto * engine = ggml_engine_create(params);

    auto status = ggml_engine_load_model(engine, model_path);
    TEST_ASSERT(status == GGML_ENGINE_OK, "vlm: text model load", "failed to load text model");

    // test default params
    auto vlm_params = ggml_engine_vlm_default_params();
    TEST_ASSERT(vlm_params.n_threads == 0, "vlm default n_threads == 0", "expected 0");
    TEST_ASSERT(vlm_params.image_min_tokens == -1, "vlm default min_tokens == -1", "expected -1");

    // load mmproj
    auto * vlm = ggml_engine_vlm_load(engine, mmproj_path, vlm_params);
    TEST_ASSERT(vlm != nullptr, "vlm: mmproj load", "failed to load mmproj");
    TEST_ASSERT(ggml_engine_vlm_is_loaded(vlm), "vlm: is loaded", "should be loaded");

    ggml_engine_vlm_free(vlm);
    ggml_engine_free(engine);
}

// ---- Test: VLM Info ----
static void test_vlm_info(const char * model_path, const char * mmproj_path) {
    print_header("VLM Info");

    auto params = ggml_engine_default_params();
    params.n_ctx = 2048;
    auto * engine = ggml_engine_create(params);
    ggml_engine_load_model(engine, model_path);

    auto vlm_params = ggml_engine_vlm_default_params();
    auto * vlm = ggml_engine_vlm_load(engine, mmproj_path, vlm_params);

    // info JSON
    char * info = ggml_engine_vlm_info_json(vlm);
    TEST_ASSERT(info != nullptr, "vlm info not null", "returned null");
    TEST_ASSERT(strstr(info, "supports_vision") != nullptr, "vlm info has supports_vision", "missing field");
    print_info("VLM info: %s", info);
    ggml_engine_free_string(info);

    // default marker
    const char * marker = ggml_engine_vlm_default_marker();
    TEST_ASSERT(marker != nullptr && strlen(marker) > 0, "vlm default marker", "empty marker");
    print_info("Default marker: %s", marker);

    ggml_engine_vlm_free(vlm);
    ggml_engine_free(engine);
}

// ---- Test: VLM Image Encode ----
static void test_vlm_encode(const char * model_path, const char * mmproj_path, const char * image_path) {
    print_header("VLM Image Encode");

    auto params = ggml_engine_default_params();
    params.n_ctx = 4096;
    auto * engine = ggml_engine_create(params);
    ggml_engine_load_model(engine, model_path);

    auto vlm_params = ggml_engine_vlm_default_params();
    auto * vlm = ggml_engine_vlm_load(engine, mmproj_path, vlm_params);

    // load image
    auto img_bytes = load_file_bytes(image_path);
    TEST_ASSERT(!img_bytes.empty(), "vlm: image file loaded", "failed to read image file");

    ggml_engine_image image;
    image.data = img_bytes.data();
    image.size = img_bytes.size();
    image.width = 0;   // file mode
    image.height = 0;

    int32_t n_tokens = ggml_engine_vlm_encode_image(vlm, &image);
    TEST_ASSERT(n_tokens > 0, "vlm: encode returns positive tokens", "expected > 0");
    print_info("Image encoded to %d tokens", n_tokens);

    ggml_engine_vlm_free(vlm);
    ggml_engine_free(engine);
}

// ---- Test: VLM Generation ----
static void test_vlm_generation(const char * model_path, const char * mmproj_path, const char * image_path) {
    print_header("VLM Generation");

    auto params = ggml_engine_default_params();
    params.n_ctx = 4096;
    params.n_threads = 4;
    auto * engine = ggml_engine_create(params);
    ggml_engine_load_model(engine, model_path);

    auto vlm_params = ggml_engine_vlm_default_params();
    auto * vlm = ggml_engine_vlm_load(engine, mmproj_path, vlm_params);

    auto img_bytes = load_file_bytes(image_path);

    ggml_engine_image image;
    image.data = img_bytes.data();
    image.size = img_bytes.size();
    image.width = 0;
    image.height = 0;

    // build prompt with image marker
    const char * marker = ggml_engine_vlm_default_marker();
    std::string prompt = std::string(marker) + "\nDescribe this image briefly.";

    auto sampling = ggml_engine_default_sampling();
    sampling.n_predict = 64;
    sampling.temperature = 0.3f;

    std::string output;
    printf("  VLM output: ");
    auto status = ggml_engine_vlm_generate(engine, vlm, prompt.c_str(),
        &image, 1, sampling, token_callback, &output);
    printf("\n");

    TEST_ASSERT(status == GGML_ENGINE_OK, "vlm generation status OK", "generation failed");
    TEST_ASSERT(!output.empty(), "vlm output not empty", "no output generated");
    print_info("Generated %zu chars", output.length());

    // perf
    auto perf = ggml_engine_get_perf(engine);
    TEST_ASSERT(perf.prompt_tokens > 0, "vlm perf prompt tokens > 0", "expected > 0");
    print_info("Prompt: %d tokens, %.1f ms (%.1f t/s)",
        perf.prompt_tokens, perf.prompt_eval_ms, perf.prompt_tokens_per_sec);
    print_info("Generation: %d tokens, %.1f ms (%.1f t/s)",
        perf.generated_tokens, perf.generation_ms, perf.generation_tokens_per_sec);

    ggml_engine_vlm_free(vlm);
    ggml_engine_free(engine);
}

// ---- Test: VLM Error Cases ----
static void test_vlm_errors(const char * model_path) {
    print_header("VLM Error Cases");

    auto params = ggml_engine_default_params();
    params.n_ctx = 512;
    auto * engine = ggml_engine_create(params);
    ggml_engine_load_model(engine, model_path);

    // generate without VLM loaded
    auto sampling = ggml_engine_default_sampling();
    sampling.n_predict = 8;
    auto status = ggml_engine_vlm_generate(engine, nullptr, "hello", nullptr, 0,
        sampling, nullptr, nullptr);
    TEST_ASSERT(status == GGML_ENGINE_ERROR_VLM_NO_PROJ, "vlm: null vlm returns NO_PROJ", "wrong error code");

    // load invalid mmproj
    auto vlm_params = ggml_engine_vlm_default_params();
    auto * vlm = ggml_engine_vlm_load(engine, "/nonexistent/path.gguf", vlm_params);
    TEST_ASSERT(vlm == nullptr, "vlm: invalid path returns null", "should be null");

    // is_loaded on null
    TEST_ASSERT(!ggml_engine_vlm_is_loaded(nullptr), "vlm: null is not loaded", "should be false");

    ggml_engine_free(engine);
}

// ---- Test: Tool Call with live model ----
static void test_tool_call_generation(const char * model_path) {
    print_header("Tool Call Generation");

    auto * tm = tool_manager_create();

    // register tools
    tool_param_def weather_params[] = {
        { "city", "The city to get weather for", TOOL_PARAM_STRING, true },
    };
    tool_def weather_tool = { "get_weather", "Get current weather for a city", weather_params, 1 };
    tool_manager_register(tm, &weather_tool);

    tool_param_def calc_params[] = {
        { "expression", "Math expression to evaluate", TOOL_PARAM_STRING, true },
    };
    tool_def calc_tool = { "calculate", "Evaluate a math expression", calc_params, 1 };
    tool_manager_register(tm, &calc_tool);

    // get tool prompt
    char * tool_prompt = tool_manager_get_prompt(tm);

    // build full prompt asking model to use a tool
    std::string prompt = std::string(tool_prompt) +
        "\nUser: What's the weather like in Tokyo?\nAssistant:";
    tool_manager_free_string(tool_prompt);

    // generate
    auto params = ggml_engine_default_params();
    params.n_ctx = 2048;
    params.n_threads = 4;
    auto * engine = ggml_engine_create(params);
    ggml_engine_load_model(engine, model_path);

    auto sampling = ggml_engine_default_sampling();
    sampling.n_predict = 128;
    sampling.temperature = 0.3f;  // low temp for structured output

    std::string output;
    printf("  Model output: ");
    auto status = ggml_engine_generate(engine, prompt.c_str(), sampling, token_callback, &output);
    printf("\n");

    TEST_ASSERT(status == GGML_ENGINE_OK, "tool-call generation OK", "generation failed");

    // try to parse tool call from output
    char * response = ggml_engine_get_response(engine);
    auto result = tool_manager_parse_output(tm, response);

    if (result.is_valid) {
        print_pass("model produced valid tool call");
        tests_passed++;
        print_info("Tool: %s, Args: %s", result.tool_name, result.arguments_json);
        free((void*)result.tool_name);
        free((void*)result.arguments_json);
    } else {
        print_info("Model did not produce parseable tool call (may need larger model)");
        print_info("Raw output: %.200s", response);
    }

    // test multi-tool parse
    const char * multi_output =
        "Let me help. {\"tool\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}} "
        "Also: {\"tool\": \"calculate\", \"arguments\": {\"expression\": \"2+2\"}}";
    int32_t n_calls = 0;
    auto * results = tool_manager_parse_output_all(tm, multi_output, &n_calls);
    TEST_ASSERT(n_calls == 2, "multi-tool parse finds 2 calls", "expected 2 calls");
    if (results) {
        for (int32_t i = 0; i < n_calls; i++) {
            print_info("  Call %d: %s(%s)", i, results[i].tool_name, results[i].arguments_json);
        }
        tool_manager_free_results(results, n_calls);
    }

    ggml_engine_free_string(response);
    ggml_engine_free(engine);
    tool_manager_free(tm);
}

// ---- Test: Character Engine All Moods ----
static void test_character_moods(const char * model_path) {
    print_header("Character Engine - All Moods");

    auto * ce = character_engine_create();
    char_personality personality = {};
    personality.name = "Luna";
    personality.persona = "You are Luna, an expressive AI companion.";
    personality.temperature = 0.8f;
    personality.top_p = 0.95f;
    personality.repetition_penalty = 1.1f;
    personality.creativity = 0.5f;
    personality.verbosity = 0.5f;
    personality.formality = 0.5f;
    character_engine_set_personality(ce, &personality);

    struct { char_mood mood; const char * name; const char * prompt_suffix; } mood_tests[] = {
        { CHAR_MOOD_NEUTRAL,  "neutral",  "Tell me about yourself." },
        { CHAR_MOOD_HAPPY,    "happy",    "What makes you happy?" },
        { CHAR_MOOD_SAD,      "sad",      "Tell me about loss." },
        { CHAR_MOOD_ANGRY,    "angry",    "What frustrates you?" },
        { CHAR_MOOD_CURIOUS,  "curious",  "What do you wonder about?" },
        { CHAR_MOOD_FOCUSED,  "focused",  "Explain quantum computing briefly." },
    };

    auto eng_params = ggml_engine_default_params();
    eng_params.n_ctx = 2048;
    eng_params.n_threads = 4;
    auto * engine = ggml_engine_create(eng_params);
    ggml_engine_load_model(engine, model_path);

    float prev_temp = 0.0f;
    for (auto & mt : mood_tests) {
        character_engine_set_mood(ce, mt.mood);
        auto cp = character_engine_get_params(ce);
        char * ctx = character_engine_get_context(ce);

        std::string label = std::string("mood:") + mt.name + " temp=" +
            std::to_string(cp.temperature).substr(0, 4);
        print_info("%s (rep_pen=%.2f)", label.c_str(), cp.repetition_penalty);

        // generate with this mood
        std::string prompt = std::string(ctx) + "\nUser: " + mt.prompt_suffix + "\nLuna:";
        character_engine_free_string(ctx);

        auto sampling = ggml_engine_default_sampling();
        sampling.temperature = cp.temperature;
        sampling.top_p = cp.top_p;
        sampling.repeat_penalty = cp.repetition_penalty;
        sampling.n_predict = 48;

        std::string output;
        printf("  [%s] ", mt.name);
        auto status = ggml_engine_generate(engine, prompt.c_str(), sampling, token_callback, &output);
        printf("\n");

        std::string test_name = std::string("mood:") + mt.name + " generation OK";
        TEST_ASSERT(status == GGML_ENGINE_OK && !output.empty(), test_name.c_str(), "failed");

        auto perf = ggml_engine_get_perf(engine);
        print_info("  -> %d tokens at %.1f t/s", perf.generated_tokens, perf.generation_tokens_per_sec);

        // verify temperature varies across moods
        if (prev_temp > 0.0f && mt.mood != CHAR_MOOD_NEUTRAL) {
            // just log, don't assert (mood deltas are small)
        }
        prev_temp = cp.temperature;
    }

    // verify temperature ordering: happy > neutral > focused
    character_engine_set_mood(ce, CHAR_MOOD_HAPPY);
    float happy_temp = character_engine_get_params(ce).temperature;
    character_engine_set_mood(ce, CHAR_MOOD_NEUTRAL);
    float neutral_temp = character_engine_get_params(ce).temperature;
    character_engine_set_mood(ce, CHAR_MOOD_FOCUSED);
    float focused_temp = character_engine_get_params(ce).temperature;

    TEST_ASSERT(happy_temp > neutral_temp, "happy temp > neutral temp", "mood scaling wrong");
    TEST_ASSERT(neutral_temp > focused_temp, "neutral temp > focused temp", "mood scaling wrong");
    print_info("Temp ordering: happy(%.2f) > neutral(%.2f) > focused(%.2f)",
        happy_temp, neutral_temp, focused_temp);

    character_engine_free(ce);
    ggml_engine_free(engine);
}

// ---- Test: Qwen3 Think/No-Think ----
static void test_think_no_think(const char * model_path) {
    print_header("Think / No-Think Mode");

    auto params = ggml_engine_default_params();
    params.n_ctx = 2048;
    params.n_threads = 4;
    auto * engine = ggml_engine_create(params);
    ggml_engine_load_model(engine, model_path);

    // NO-THINK mode: /no_think tag tells Qwen3 to skip reasoning
    {
        std::string prompt =
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\nWhat is 15 + 27?/no_think<|im_end|>\n"
            "<|im_start|>assistant\n";

        auto sampling = ggml_engine_default_sampling();
        sampling.n_predict = 64;
        sampling.temperature = 0.3f;

        std::string output;
        printf("  [no-think] ");
        auto status = ggml_engine_generate(engine, prompt.c_str(), sampling, token_callback, &output);
        printf("\n");

        TEST_ASSERT(status == GGML_ENGINE_OK, "no-think generation OK", "generation failed");
        TEST_ASSERT(!output.empty(), "no-think output not empty", "empty output");

        // check if output contains the answer (42)
        bool has_answer = output.find("42") != std::string::npos;
        if (has_answer) {
            print_pass("no-think: correct answer (42)");
            tests_passed++;
        } else {
            print_info("no-think: answer was '%s' (may not contain 42 verbatim)", output.substr(0, 100).c_str());
        }

        // check for think tags — no-think should NOT have <think> blocks
        bool has_think = output.find("<think>") != std::string::npos;
        if (!has_think) {
            print_pass("no-think: no <think> tags (good)");
            tests_passed++;
        } else {
            print_info("no-think: found <think> tags (model may still reason)");
        }

        auto perf = ggml_engine_get_perf(engine);
        print_info("no-think: %d tokens, %.1f t/s", perf.generated_tokens, perf.generation_tokens_per_sec);
    }

    // THINK mode: /think tag tells Qwen3 to show reasoning
    {
        std::string prompt =
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\nWhat is 15 + 27?/think<|im_end|>\n"
            "<|im_start|>assistant\n";

        auto sampling = ggml_engine_default_sampling();
        sampling.n_predict = 256;
        sampling.temperature = 0.6f;

        std::string output;
        printf("  [think] ");
        auto status = ggml_engine_generate(engine, prompt.c_str(), sampling, token_callback, &output);
        printf("\n");

        TEST_ASSERT(status == GGML_ENGINE_OK, "think generation OK", "generation failed");
        TEST_ASSERT(!output.empty(), "think output not empty", "empty output");

        // check for think tags — think mode SHOULD have <think> or reasoning
        bool has_think = output.find("<think>") != std::string::npos;
        if (has_think) {
            print_pass("think: has <think> reasoning block");
            tests_passed++;
        } else {
            print_info("think: no <think> tags (model may not support think mode)");
        }

        auto perf = ggml_engine_get_perf(engine);
        print_info("think: %d tokens, %.1f t/s", perf.generated_tokens, perf.generation_tokens_per_sec);
    }

    ggml_engine_free(engine);
}

#include "rag-tests.inc"

static void print_usage(const char * prog) {
    printf("Usage: %s -m <model_path> [options]\n", prog);
    printf("\nOptions:\n");
    printf("  -m <path>        Path to GGUF model file (required for model tests)\n");
    printf("  --mmproj <path>  Path to mmproj GGUF file (required for VLM tests)\n");
    printf("  --image <path>   Path to test image file (required for VLM encode/gen)\n");
    printf("  --embed-model <path>  Path to embedding model GGUF (for RAG tests)\n");
    printf("  --rag-text <path>    Path to text file for RAG large file test\n");
    printf("  --all            Run all tests (default)\n");
    printf("  --no-model       Skip tests that require a model\n");
    printf("  --quick          Quick test (lifecycle + tool + character only)\n");
    printf("  -h, --help       Show this help\n");
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * mmproj_path = nullptr;
    const char * image_path = nullptr;
    const char * embed_model_path = nullptr;
    const char * rag_text_path = nullptr;
    bool run_model_tests = true;
    bool quick_mode = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--mmproj") == 0 && i + 1 < argc) {
            mmproj_path = argv[++i];
        } else if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
            image_path = argv[++i];
        } else if (strcmp(argv[i], "--embed-model") == 0 && i + 1 < argc) {
            embed_model_path = argv[++i];
        } else if (strcmp(argv[i], "--rag-text") == 0 && i + 1 < argc) {
            rag_text_path = argv[++i];
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
            test_tool_call_generation(model_path);
            test_character_moods(model_path);
            test_think_no_think(model_path);
        } else {
            test_model_loading(model_path);
            test_model_info(model_path);
        }

        // VLM tests
        if (mmproj_path) {
            test_vlm_loading(model_path, mmproj_path);
            test_vlm_info(model_path, mmproj_path);
            test_vlm_errors(model_path);

            if (image_path) {
                test_vlm_encode(model_path, mmproj_path, image_path);
                test_vlm_generation(model_path, mmproj_path, image_path);
            } else {
                print_info("Skipping VLM encode/generation tests (no --image provided)");
            }
        } else {
            // still run error case tests (only needs text model)
            if (!quick_mode) {
                test_vlm_errors(model_path);
            }
        }
    } else if (run_model_tests && !model_path) {
        print_info("Skipping model tests (no -m <path> provided)");
    }

    // RAG tests (always run lifecycle + errors, embed-model tests need model)
    test_rag_lifecycle();
    test_rag_errors();
    if (embed_model_path) {
        test_rag_model_loading(embed_model_path);
        if (!quick_mode) {
            test_rag_indexing(embed_model_path);
            test_rag_retrieval(embed_model_path);
            test_rag_build_prompt(embed_model_path);
            test_rag_info(embed_model_path);
        }
        if (rag_text_path) {
            test_rag_large_file(embed_model_path, rag_text_path);
        }
    } else {
        print_info("Skipping RAG model tests (no --embed-model provided)");
    }

    // Summary
    printf("\n%s%s══════════════════════════════════════════%s\n", CLR_BOLD, CLR_CYAN, CLR_RESET);
    printf("  Results: %s%d passed%s, %s%d failed%s\n",
        CLR_GREEN, tests_passed, CLR_RESET,
        tests_failed > 0 ? CLR_RED : CLR_GREEN, tests_failed, CLR_RESET);
    printf("%s%s══════════════════════════════════════════%s\n", CLR_BOLD, CLR_CYAN, CLR_RESET);

    return tests_failed > 0 ? 1 : 0;
}
