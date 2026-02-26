cmake_minimum_required(VERSION 3.14)
project(amx-server CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ggml library from parent project
set(GGML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml)

# Build ggml as subdirectory (host build, no Android toolchain)
add_subdirectory(${GGML_DIR} ${CMAKE_BINARY_DIR}/ggml)

# Server executable
add_executable(amx-server
    src/main.cpp
    src/gguf_parser.cpp
    src/adb_bridge.cpp
    src/graph_builder.cpp
    src/amxp_format.cpp
)

target_include_directories(amx-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor
    ${GGML_DIR}/include
)

target_link_libraries(amx-server PRIVATE ggml ggml-base)

# Threading for cpp-httplib
find_package(Threads REQUIRED)
target_link_libraries(amx-server PRIVATE Threads::Threads)
