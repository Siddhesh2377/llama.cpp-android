// RAG Engine tests — included by llama-test-cli.cpp

// ---- Test: RAG Engine lifecycle ----
static void test_rag_lifecycle() {
    print_header("RAG Lifecycle");

    auto params = rag_engine_default_params();
    TEST_ASSERT(params.chunk_size == 256, "rag default chunk_size", "expected 256");
    TEST_ASSERT(params.n_dims == 256, "rag default n_dims", "expected 256");
    TEST_ASSERT(params.top_k == 32, "rag default top_k", "expected 32");
    TEST_ASSERT(params.top_n == 5, "rag default top_n", "expected 5");
    TEST_ASSERT(params.late_chunking == true, "rag default late_chunking", "expected true");

    auto * rag = rag_engine_create(params);
    TEST_ASSERT(rag != nullptr, "rag create", "returned null");
    TEST_ASSERT(!rag_engine_is_loaded(rag), "rag not loaded initially", "should not be loaded");
    TEST_ASSERT(rag_engine_document_count(rag) == 0, "rag no docs", "expected 0");
    TEST_ASSERT(rag_engine_chunk_count(rag) == 0, "rag no chunks", "expected 0");

    rag_engine_free(rag);
    print_pass("rag free");
    tests_passed++;
}

// ---- Test: RAG Model loading ----
static void test_rag_model_loading(const char * embed_model_path) {
    print_header("RAG Model Loading");

    auto params = rag_engine_default_params();
    auto * rag = rag_engine_create(params);

    int32_t r = rag_engine_load_model(rag, embed_model_path);
    TEST_ASSERT(r == 0, "rag model load", "failed to load model");
    TEST_ASSERT(rag_engine_is_loaded(rag), "rag model is loaded", "should be loaded");

    char * info = rag_engine_info_json(rag);
    TEST_ASSERT(info != nullptr, "rag info json", "returned null");
    if (info) {
        print_info("RAG info: %s", info);
        rag_engine_free_string(info);
    }

    rag_engine_free(rag);
}

// ---- Test: RAG Indexing ----
static void test_rag_indexing(const char * embed_model_path) {
    print_header("RAG Indexing");

    auto params = rag_engine_default_params();
    auto * rag = rag_engine_create(params);
    rag_engine_load_model(rag, embed_model_path);

    const char * doc1 = "Albert Einstein developed the theory of general relativity. "
                        "He received the Nobel Prize in Physics in 1921 for his discovery "
                        "of the law of the photoelectric effect.";
    const char * doc2 = "Marie Curie was a physicist and chemist who conducted pioneering "
                        "research on radioactivity. She was the first woman to win a Nobel Prize.";

    int32_t n1 = rag_engine_add_document(rag, doc1, "einstein");
    TEST_ASSERT(n1 > 0, "rag add doc einstein", "expected >0 chunks");
    print_info("einstein: %d chunks", n1);

    int32_t n2 = rag_engine_add_document(rag, doc2, "curie");
    TEST_ASSERT(n2 > 0, "rag add doc curie", "expected >0 chunks");
    print_info("curie: %d chunks", n2);

    TEST_ASSERT(rag_engine_document_count(rag) == 2, "rag doc count", "expected 2");
    TEST_ASSERT(rag_engine_chunk_count(rag) == n1 + n2, "rag chunk count", "mismatch");

    int32_t r = rag_engine_remove_document(rag, "einstein");
    TEST_ASSERT(r == 0, "rag remove doc einstein", "failed");
    TEST_ASSERT(rag_engine_document_count(rag) == 1, "rag doc count after remove", "expected 1");

    rag_engine_clear(rag);
    TEST_ASSERT(rag_engine_chunk_count(rag) == 0, "rag chunks after clear", "expected 0");

    rag_engine_free(rag);
}

// ---- Test: RAG Retrieval ----
static void test_rag_retrieval(const char * embed_model_path) {
    print_header("RAG Retrieval");

    auto params = rag_engine_default_params();
    params.top_n = 2;
    auto * rag = rag_engine_create(params);
    rag_engine_load_model(rag, embed_model_path);

    rag_engine_add_document(rag,
        "The mitochondria is the powerhouse of the cell. It generates ATP through "
        "oxidative phosphorylation. Mitochondria have their own DNA.",
        "biology");
    rag_engine_add_document(rag,
        "Python is a programming language created by Guido van Rossum. It is known "
        "for its simple syntax and readability.",
        "python");
    rag_engine_add_document(rag,
        "The Great Wall of China stretches over 13,000 miles. It was built over "
        "many centuries to protect against invasions from the north.",
        "history");

    int32_t n_results = 0;
    auto * results = rag_engine_query(rag, "What generates energy in cells?", &n_results);
    TEST_ASSERT(n_results > 0, "rag query returns results", "no results");
    if (n_results > 0 && results) {
        print_info("Top result: doc=%s score=%.4f", results[0].doc_id, results[0].score);
        TEST_ASSERT(strcmp(results[0].doc_id, "biology") == 0,
            "rag top result is biology", "wrong doc");
        if (n_results > 1) {
            TEST_ASSERT(results[0].score >= results[1].score,
                "rag scores descending", "wrong order");
        }
        rag_engine_free_results(results, n_results);
    }

    rag_engine_free(rag);
}

// ---- Test: RAG Build Prompt ----
static void test_rag_build_prompt(const char * embed_model_path) {
    print_header("RAG Build Prompt");

    auto params = rag_engine_default_params();
    params.top_n = 1;
    auto * rag = rag_engine_create(params);
    rag_engine_load_model(rag, embed_model_path);

    rag_engine_add_document(rag,
        "The speed of light is 299,792,458 meters per second.", "physics");

    char * prompt = rag_engine_build_prompt(rag,
        "How fast is light?", "Tell me about the speed of light.");
    TEST_ASSERT(prompt != nullptr, "rag build prompt", "returned null");
    if (prompt) {
        TEST_ASSERT(strstr(prompt, "Context:") != nullptr, "rag prompt has context", "missing");
        TEST_ASSERT(strstr(prompt, "speed of light") != nullptr, "rag prompt has content", "missing");
        TEST_ASSERT(strstr(prompt, "Tell me about") != nullptr, "rag prompt has user text", "missing");
        print_info("Prompt length: %zu chars", strlen(prompt));
        rag_engine_free_string(prompt);
    }

    rag_engine_free(rag);
}

// ---- Test: RAG Info ----
static void test_rag_info(const char * embed_model_path) {
    print_header("RAG Info");

    auto params = rag_engine_default_params();
    auto * rag = rag_engine_create(params);
    rag_engine_load_model(rag, embed_model_path);

    rag_engine_add_document(rag, "Some test content for info.", "info-test");

    char * info = rag_engine_info_json(rag);
    TEST_ASSERT(info != nullptr, "rag info not null", "returned null");
    if (info) {
        TEST_ASSERT(strstr(info, "\"loaded\":true") != nullptr, "rag info loaded", "missing");
        TEST_ASSERT(strstr(info, "\"n_documents\":1") != nullptr, "rag info 1 doc", "missing");
        TEST_ASSERT(strstr(info, "\"n_dims\":256") != nullptr, "rag info dims", "missing");
        print_info("Info: %s", info);
        rag_engine_free_string(info);
    }

    rag_engine_free(rag);
}

// ---- Test: RAG Error cases ----
static void test_rag_errors() {
    print_header("RAG Error Cases");

    TEST_ASSERT(rag_engine_is_loaded(nullptr) == false, "rag null not loaded", "should be false");
    TEST_ASSERT(rag_engine_document_count(nullptr) == 0, "rag null 0 docs", "should be 0");

    auto params = rag_engine_default_params();
    auto * rag = rag_engine_create(params);

    TEST_ASSERT(rag_engine_add_document(rag, "text", "id") == -1,
        "rag add doc without model", "should fail");
    int32_t n = 0;
    TEST_ASSERT(rag_engine_query(rag, "query", &n) == nullptr,
        "rag query without model", "should fail");
    TEST_ASSERT(rag_engine_load_model(rag, "/nonexistent/model.gguf") == -1,
        "rag bad model path", "should fail");

    rag_engine_free(rag);
}

// ---- Test: RAG with large text file ----
static void test_rag_large_file(const char * embed_model_path, const char * text_file_path) {
    print_header("RAG Large File Test");

    // Load text file
    std::ifstream f(text_file_path, std::ios::ate);
    if (!f.is_open()) {
        print_fail("rag load text file", "failed to open");
        tests_failed++;
        return;
    }
    size_t file_size = f.tellg();
    f.seekg(0);
    std::string text(file_size, '\0');
    f.read(&text[0], file_size);
    f.close();

    print_info("Text file: %s (%zu bytes, ~%zu KB)", text_file_path, file_size, file_size / 1024);

    auto params = rag_engine_default_params();
    params.top_n = 3;
    auto * rag = rag_engine_create(params);

    int32_t r = rag_engine_load_model(rag, embed_model_path);
    TEST_ASSERT(r == 0, "rag large: model load", "failed");
    if (r != 0) { rag_engine_free(rag); return; }

    // Index the full text as one document
    print_info("Indexing document (late chunking)...");
    auto t_start = std::chrono::high_resolution_clock::now();

    int32_t n_chunks = rag_engine_add_document(rag, text.c_str(), "large-text");

    auto t_end = std::chrono::high_resolution_clock::now();
    double index_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    TEST_ASSERT(n_chunks > 0, "rag large: indexing produced chunks", "no chunks");
    print_info("Indexed into %d chunks in %.1f ms (%.1f ms/chunk)",
        n_chunks, index_ms, n_chunks > 0 ? index_ms / n_chunks : 0.0);

    // Query 1: specific factual query
    {
        int32_t n_results = 0;
        auto t_q = std::chrono::high_resolution_clock::now();
        auto * results = rag_engine_query(rag, "Napoleon and the battle", &n_results);
        auto t_qe = std::chrono::high_resolution_clock::now();
        double query_ms = std::chrono::duration<double, std::milli>(t_qe - t_q).count();

        TEST_ASSERT(n_results > 0, "rag large: query returns results", "no results");
        if (n_results > 0 && results) {
            print_info("Query 'Napoleon and the battle': %d results in %.1f ms", n_results, query_ms);
            for (int32_t i = 0; i < n_results; i++) {
                print_info("  [%d] score=%.4f chunk=%d text=%.80s...",
                    i, results[i].score, results[i].chunk_index,
                    results[i].text);
            }
            rag_engine_free_results(results, n_results);
        }
    }

    // Query 2: different topic
    {
        int32_t n_results = 0;
        auto t_q = std::chrono::high_resolution_clock::now();
        auto * results = rag_engine_query(rag, "love and marriage", &n_results);
        auto t_qe = std::chrono::high_resolution_clock::now();
        double query_ms = std::chrono::duration<double, std::milli>(t_qe - t_q).count();

        TEST_ASSERT(n_results > 0, "rag large: query 2 returns results", "no results");
        if (n_results > 0 && results) {
            print_info("Query 'love and marriage': %d results in %.1f ms", n_results, query_ms);
            for (int32_t i = 0; i < n_results; i++) {
                print_info("  [%d] score=%.4f chunk=%d text=%.80s...",
                    i, results[i].score, results[i].chunk_index,
                    results[i].text);
            }
            rag_engine_free_results(results, n_results);
        }
    }

    // Query 3: build augmented prompt
    {
        char * prompt = rag_engine_build_prompt(rag,
            "What happened at the battle of Borodino?",
            "Tell me about the battle of Borodino.");
        TEST_ASSERT(prompt != nullptr, "rag large: build prompt", "null");
        if (prompt) {
            print_info("Built prompt: %zu chars", strlen(prompt));
            rag_engine_free_string(prompt);
        }
    }

    // Info
    char * info = rag_engine_info_json(rag);
    if (info) {
        print_info("Engine info: %s", info);
        rag_engine_free_string(info);
    }

    rag_engine_free(rag);
}
