# gguf-engine SDK — Modular C++ inference engine for Android
#
# Targets:
#   gguf-engine      — static library (all modules)
#   gguf-engine-cli  — standalone CLI executable
#   ggml-test        — GPU/CPU benchmark test (legacy)
#   gguf-forward-test — monolithic engine (legacy, kept for reference)

# ---------------------------------------------------------------------------
# SDK static library
# ---------------------------------------------------------------------------
set(ENGINE_SOURCES
    src/model.cpp
    src/graph.cpp
    src/interventions.cpp
    src/sampling.cpp
    src/tokenizer.cpp
    src/personality.cpp
    src/emotional.cpp
    src/grammar.cpp
    src/boundaries.cpp
    src/rag.cpp
    src/stall.cpp
    src/vision.cpp
    src/tui.cpp
    src/chat.cpp
)

add_library(gguf-engine STATIC ${ENGINE_SOURCES})

target_include_directories(gguf-engine
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/ggml/include
)

target_link_libraries(gguf-engine PUBLIC ggml ggml-base)

# C++17 for inline variables in headers
target_compile_features(gguf-engine PUBLIC cxx_std_17)

# ---------------------------------------------------------------------------
# CLI executable (modular SDK)
# ---------------------------------------------------------------------------
add_executable(gguf-engine-cli
    main.cpp
    tests/test_engine.cpp
)

target_link_libraries(gguf-engine-cli PRIVATE gguf-engine)

target_include_directories(gguf-engine-cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
)

# ---------------------------------------------------------------------------
# Legacy targets (kept for reference/compatibility)
# ---------------------------------------------------------------------------
add_executable(ggml-test ggml-test.cpp)
target_link_libraries(ggml-test PRIVATE ggml ggml-base)
target_include_directories(ggml-test PRIVATE ${CMAKE_SOURCE_DIR}/ggml/include)

add_executable(gguf-forward-test gguf-forward-test.cpp)
target_link_libraries(gguf-forward-test PRIVATE ggml ggml-base)
target_include_directories(gguf-forward-test PRIVATE ${CMAKE_SOURCE_DIR}/ggml/include)
