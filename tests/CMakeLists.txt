llama_add_compile_flags()

add_executable(test-universal-toolcall test-universal-toolcall.cpp)
target_link_libraries(test-universal-toolcall PRIVATE common llama)
target_compile_features(test-universal-toolcall PRIVATE cxx_std_17)

add_test(NAME test-universal-toolcall COMMAND test-universal-toolcall)
