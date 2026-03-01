# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

add_subdirectory(quantize)
add_subdirectory(completion)
if (LLAMA_BUILD_SERVER)
    add_subdirectory(cli)
    add_subdirectory(server)
endif()
