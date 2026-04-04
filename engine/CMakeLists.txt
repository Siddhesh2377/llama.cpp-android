if(TARGET tn-engine)
    return()
endif()

set(VLM_SOURCES
    vlm/clip.cpp
    vlm/mtmd.cpp
    vlm/mtmd-helper.cpp
    vlm/mtmd-audio.cpp
    vlm/models/cogvlm.cpp
    vlm/models/conformer.cpp
    vlm/models/glm4v.cpp
    vlm/models/internvl.cpp
    vlm/models/kimik25.cpp
    vlm/models/kimivl.cpp
    vlm/models/llama4.cpp
    vlm/models/llava.cpp
    vlm/models/minicpmv.cpp
    vlm/models/mobilenetv5.cpp
    vlm/models/nemotron-v2-vl.cpp
    vlm/models/paddleocr.cpp
    vlm/models/pixtral.cpp
    vlm/models/qwen2vl.cpp
    vlm/models/qwen3vl.cpp
    vlm/models/siglip.cpp
    vlm/models/whisper-enc.cpp
    vlm/models/youtuvl.cpp
)

set(ENGINE_SOURCES
    tn-log.cpp
    ggml-engine.cpp
    ggml-engine-vlm.cpp
    rag-engine.cpp
    tool-manager.cpp
    ${VLM_SOURCES}
)

add_library(tn-engine STATIC ${ENGINE_SOURCES})

target_include_directories(tn-engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/vlm
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/vendor
)

target_link_libraries(tn-engine PUBLIC llama common ggml)
target_compile_features(tn-engine PRIVATE cxx_std_17)

target_compile_options(tn-engine PRIVATE -Wno-cast-qual)
