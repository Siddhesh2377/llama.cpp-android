cmake_minimum_required(VERSION 3.14)
project("ggml" C CXX ASM)

### GGML Version
set(GGML_VERSION_MAJOR 0)
set(GGML_VERSION_MINOR 9)
set(GGML_VERSION_PATCH 5)
set(GGML_VERSION_BASE "${GGML_VERSION_MAJOR}.${GGML_VERSION_MINOR}.${GGML_VERSION_PATCH}")
set(GGML_VERSION "${GGML_VERSION_BASE}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(GGML_STANDALONE ON)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
else()
    set(GGML_STANDALONE OFF)
    if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    endif()
endif()

set(BUILD_SHARED_LIBS_DEFAULT ON)
option(BUILD_SHARED_LIBS           "ggml: build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})
option(GGML_BACKEND_DL             "ggml: build backends as dynamic libraries" OFF)
set(GGML_BACKEND_DIR "" CACHE PATH "ggml: directory to load dynamic backends from")

# general
option(GGML_STATIC  "ggml: static link libraries"                     OFF)
option(GGML_NATIVE  "ggml: optimize the build for the current system" OFF)
option(GGML_LTO     "ggml: enable link time optimization"             OFF)
option(GGML_CCACHE  "ggml: use ccache if available"                   ON)

# debug
option(GGML_ALL_WARNINGS  "ggml: enable all compiler warnings" ON)
option(GGML_GPROF         "ggml: enable gprof"                 OFF)
option(GGML_FATAL_WARNINGS "ggml: enable -Werror flag"         OFF)

# sanitizers
option(GGML_SANITIZE_THREAD    "ggml: enable thread sanitizer"    OFF)
option(GGML_SANITIZE_ADDRESS   "ggml: enable address sanitizer"   OFF)
option(GGML_SANITIZE_UNDEFINED "ggml: enable undefined sanitizer" OFF)

# CPU options (ARM-focused)
option(GGML_CPU_REPACK       "ggml: runtime weight conversion of Q4_0 to Q4_X_X" ON)
option(GGML_CPU_ALL_VARIANTS "ggml: build all ARM CPU variants (requires GGML_BACKEND_DL)" OFF)
set(GGML_CPU_ARM_ARCH        "" CACHE STRING "ggml: CPU architecture for ARM")

# ggml core
set(GGML_SCHED_MAX_COPIES  "4" CACHE STRING "ggml: max input copies for pipeline parallelism")
option(GGML_CPU             "ggml: enable CPU backend"     ON)
option(GGML_SCHED_NO_REALLOC "ggml: disallow reallocations in ggml-alloc" OFF)

# mobile backends
option(GGML_OPENMP   "ggml: use OpenMP"    ON)
option(GGML_OPENCL   "ggml: use OpenCL"    OFF)

option(GGML_OPENCL_PROFILING          "ggml: use OpenCL profiling"        OFF)
option(GGML_OPENCL_EMBED_KERNELS      "ggml: embed OpenCL kernels"        ON)
option(GGML_OPENCL_USE_ADRENO_KERNELS "ggml: optimized kernels for Adreno" ON)
set(GGML_OPENCL_TARGET_VERSION "300" CACHE STRING "ggml: OpenCL API version")

# x86 instruction sets (for Linux host build/testing only)
if (NOT CMAKE_CROSSCOMPILING)
    set(INS_ENB ON)
else()
    set(INS_ENB OFF)
endif()
option(GGML_SSE42   "ggml: enable SSE 4.2"  ${INS_ENB})
option(GGML_AVX     "ggml: enable AVX"       ${INS_ENB})
option(GGML_AVX2    "ggml: enable AVX2"      ${INS_ENB})
option(GGML_BMI2    "ggml: enable BMI2"      ${INS_ENB})
option(GGML_FMA     "ggml: enable FMA"       ${INS_ENB})
option(GGML_F16C    "ggml: enable F16C"      ${INS_ENB})
option(GGML_AVX512  "ggml: enable AVX512F"   OFF)
option(GGML_AVX_VNNI "ggml: enable AVX-VNNI" OFF)
option(GGML_AVX512_VBMI "ggml: enable AVX512-VBMI" OFF)
option(GGML_AVX512_VNNI "ggml: enable AVX512-VNNI" OFF)
option(GGML_AVX512_BF16 "ggml: enable AVX512-BF16" OFF)
option(GGML_AMX_TILE "ggml: enable AMX-TILE" OFF)
option(GGML_AMX_INT8 "ggml: enable AMX-INT8" OFF)
option(GGML_AMX_BF16 "ggml: enable AMX-BF16" OFF)

# llamafile (disabled on Android)
option(GGML_LLAMAFILE "ggml: use LLAMAFILE" OFF)

# KleidiAI (disabled - requires newer cmake)
option(GGML_CPU_KLEIDIAI "ggml: use KleidiAI" OFF)
option(GGML_CPU_HBM      "ggml: use memkind for CPU HBM" OFF)

# dependencies
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

include(GNUInstallDirs)

# build
add_subdirectory(src)

# install
include(CMakePackageConfigHelpers)

set(GGML_PUBLIC_HEADERS
    include/ggml.h
    include/ggml-cpu.h
    include/ggml-alloc.h
    include/ggml-backend.h
    include/ggml-cpp.h
    include/ggml-opt.h
    include/gguf.h)

set_target_properties(ggml PROPERTIES PUBLIC_HEADER "${GGML_PUBLIC_HEADERS}")
install(TARGETS ggml LIBRARY PUBLIC_HEADER)
install(TARGETS ggml-base LIBRARY)

set(GGML_INSTALL_VERSION ${GGML_VERSION})
set(GGML_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Header files")
set(GGML_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Library files")
set(GGML_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Binary files")

target_compile_definitions(ggml-base PRIVATE
    GGML_VERSION="${GGML_INSTALL_VERSION}"
    GGML_COMMIT="mobile"
)
message(STATUS "ggml version: ${GGML_INSTALL_VERSION}")
