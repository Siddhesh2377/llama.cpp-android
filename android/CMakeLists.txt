add_executable(ggml-test ggml-test.cpp)
target_link_libraries(ggml-test PRIVATE ggml ggml-base)
target_include_directories(ggml-test PRIVATE ${CMAKE_SOURCE_DIR}/ggml/include)

add_executable(gguf-forward-test gguf-forward-test.cpp)
target_link_libraries(gguf-forward-test PRIVATE ggml ggml-base)
target_include_directories(gguf-forward-test PRIVATE ${CMAKE_SOURCE_DIR}/ggml/include)
