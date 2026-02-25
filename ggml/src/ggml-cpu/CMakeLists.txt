function(ggml_add_cpu_backend_features cpu_name arch)
    set(GGML_CPU_FEATS_NAME ${cpu_name}-feats)
    add_library(${GGML_CPU_FEATS_NAME} OBJECT ggml-cpu/arch/${arch}/cpu-feats.cpp)
    target_include_directories(${GGML_CPU_FEATS_NAME} PRIVATE . ../include)
    target_compile_definitions(${GGML_CPU_FEATS_NAME} PRIVATE ${ARGN})
    target_compile_definitions(${GGML_CPU_FEATS_NAME} PRIVATE GGML_BACKEND_DL GGML_BACKEND_BUILD GGML_BACKEND_SHARED)
    set_target_properties(${GGML_CPU_FEATS_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${cpu_name} PRIVATE ${GGML_CPU_FEATS_NAME})
endfunction()

function(ggml_add_cpu_backend_variant_impl tag_name)
    if (tag_name)
        set(GGML_CPU_NAME ggml-cpu-${tag_name})
    else()
        set(GGML_CPU_NAME ggml-cpu)
    endif()

    ggml_add_backend_library(${GGML_CPU_NAME})

    list(APPEND GGML_CPU_SOURCES
        ggml-cpu/ggml-cpu.c
        ggml-cpu/ggml-cpu.cpp
        ggml-cpu/repack.cpp
        ggml-cpu/repack.h
        ggml-cpu/quants.c
        ggml-cpu/quants.h
        ggml-cpu/traits.cpp
        ggml-cpu/traits.h
        ggml-cpu/ggml-cpu-impl.h
        ggml-cpu/common.h
        ggml-cpu/binary-ops.h
        ggml-cpu/binary-ops.cpp
        ggml-cpu/unary-ops.h
        ggml-cpu/unary-ops.cpp
        ggml-cpu/simd-mappings.h
        ggml-cpu/vec.h
        ggml-cpu/vec.cpp
        ggml-cpu/ops.h
        ggml-cpu/ops.cpp
    )

    target_compile_features(${GGML_CPU_NAME} PRIVATE c_std_11 cxx_std_17)
    target_include_directories(${GGML_CPU_NAME} PRIVATE . ggml-cpu)

    if (GGML_OPENMP)
        find_package(OpenMP)
        if (OpenMP_FOUND)
            set(GGML_OPENMP_ENABLED "ON" CACHE INTERNAL "")
            target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_OPENMP)
            target_link_libraries(${GGML_CPU_NAME} PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
        else()
            set(GGML_OPENMP_ENABLED "OFF" CACHE INTERNAL "")
            message(WARNING "OpenMP not found")
        endif()
    endif()

    # ARM architecture (Android + Linux ARM)
    if (GGML_SYSTEM_ARCH STREQUAL "ARM")
        message(STATUS "ARM detected")
        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/arch/arm/quants.c
            ggml-cpu/arch/arm/repack.cpp
        )

        check_cxx_compiler_flag(-mfp16-format=ieee GGML_COMPILER_SUPPORTS_FP16_FORMAT_I3E)
        if (NOT "${GGML_COMPILER_SUPPORTS_FP16_FORMAT_I3E}" STREQUAL "")
            list(APPEND ARCH_FLAGS -mfp16-format=ieee)
        endif()

        if (GGML_NATIVE)
            list(APPEND ARCH_FLAGS -mcpu=native)
        elseif (GGML_CPU_ARM_ARCH)
            list(APPEND ARCH_FLAGS -march=${GGML_CPU_ARM_ARCH})
        elseif (GGML_CPU_ALL_VARIANTS)
            # Build per-variant with specific feature sets
            set(ARM_MCPU "armv8-a")
            set(ARCH_TAGS "")
            set(ARCH_DEFINITIONS "")

            if (GGML_INTERNAL_DOTPROD)
                set(ARM_MCPU "armv8.2-a")
                set(ARCH_TAGS "${ARCH_TAGS}+dotprod")
                list(APPEND ARCH_DEFINITIONS GGML_USE_DOTPROD)
            endif()
            if (GGML_INTERNAL_FP16_VECTOR_ARITHMETIC)
                set(ARM_MCPU "armv8.2-a")
                set(ARCH_TAGS "${ARCH_TAGS}+fp16")
                list(APPEND ARCH_DEFINITIONS GGML_USE_FP16_VECTOR_ARITHMETIC)
            endif()
            if (GGML_INTERNAL_SVE)
                set(ARM_MCPU "armv8.2-a")
                set(ARCH_TAGS "${ARCH_TAGS}+sve")
                list(APPEND ARCH_DEFINITIONS GGML_USE_SVE)
            endif()
            if (GGML_INTERNAL_MATMUL_INT8)
                set(ARM_MCPU "armv8.6-a")
                set(ARCH_TAGS "${ARCH_TAGS}+i8mm")
                list(APPEND ARCH_DEFINITIONS GGML_USE_MATMUL_INT8)
            endif()
            if (GGML_INTERNAL_SVE2)
                set(ARM_MCPU "armv8.6-a")
                set(ARCH_TAGS "${ARCH_TAGS}+sve2")
                list(APPEND ARCH_DEFINITIONS GGML_USE_SVE2)
            endif()
            if (GGML_INTERNAL_NOSVE)
                set(ARCH_TAGS "${ARCH_TAGS}+nosve")
            endif()
            if (GGML_INTERNAL_SME)
                set(ARM_MCPU "armv9.2-a")
                set(ARCH_TAGS "${ARCH_TAGS}+sme")
                list(APPEND ARCH_DEFINITIONS GGML_USE_SME)
            endif()
            list(APPEND ARCH_FLAGS "-march=${ARM_MCPU}${ARCH_TAGS}")
            ggml_add_cpu_backend_features(${GGML_CPU_NAME} arm ${ARCH_DEFINITIONS})
        endif()

        # Feature detection
        if (ARCH_FLAGS)
            include(CheckCXXSourceCompiles)
            set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
            string(REPLACE ";" " " ARCH_FLAGS_STR "${ARCH_FLAGS}")
            set(CMAKE_REQUIRED_FLAGS "${ARCH_FLAGS_STR}")
            foreach(feature DOTPROD SVE MATMUL_INT8 FMA FP16_VECTOR_ARITHMETIC SME)
                check_cxx_source_compiles(
                    "
                    #if !defined(__ARM_FEATURE_${feature})
                    #  error \"Feature ${feature} is not defined\"
                    #endif
                    int main() { return 0; }
                    "
                    HAVE_${feature}
                )
            endforeach()
            set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})
        endif()
    else()
        message(WARNING "Non-ARM architecture detected (${GGML_SYSTEM_ARCH}). This build is optimized for ARM.")
        list(APPEND ARCH_FLAGS -DGGML_CPU_GENERIC)
    endif()

    if (GGML_CPU_REPACK)
        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_CPU_REPACK)
    endif()

    message(STATUS "Adding CPU backend variant ${GGML_CPU_NAME}: ${ARCH_FLAGS} ${ARCH_DEFINITIONS}")
    target_sources(${GGML_CPU_NAME} PRIVATE ${GGML_CPU_SOURCES})
    target_compile_options(${GGML_CPU_NAME} PRIVATE ${ARCH_FLAGS})
    target_compile_definitions(${GGML_CPU_NAME} PRIVATE ${ARCH_DEFINITIONS})
endfunction()
