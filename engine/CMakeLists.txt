set(ENGINE_SOURCES
    ggml-engine.cpp
    tool-manager.cpp
    character-engine.cpp
    kv-cache-manager.cpp
)

add_library(tn-engine STATIC ${ENGINE_SOURCES})

target_include_directories(tn-engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/ggml/include
)

target_link_libraries(tn-engine PUBLIC llama common)
target_compile_features(tn-engine PRIVATE cxx_std_17)

# Test CLI
add_executable(llama-test-cli llama-test-cli.cpp)
target_link_libraries(llama-test-cli PRIVATE tn-engine llama common)
target_compile_features(llama-test-cli PRIVATE cxx_std_17)
install(TARGETS llama-test-cli RUNTIME)
