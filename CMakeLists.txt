cmake_minimum_required(VERSION 3.20)

project(ncnn_llm VERSION 0.1.0 LANGUAGES C CXX)

option(NCNN_LLM_BUILD_OCR "Build the OCR runtime and command-line example" ON)
option(NCNN_LLM_BUILD_TESTS "Build model-independent unit tests" ON)
option(NCNN_LLM_FETCH_DEPS "Fetch ncnn and nlohmann_json when packages are unavailable" OFF)
set(NCNN_SOURCE_DIR "" CACHE PATH "Path to an ncnn source checkout (optional)")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/utf-8)
    add_compile_definitions(NOMINMAX)
endif()

if(NCNN_SOURCE_DIR)
    set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(NCNN_VULKAN OFF CACHE BOOL "" FORCE)
    add_subdirectory("${NCNN_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/ncnn")
else()
    find_package(ncnn CONFIG QUIET)
endif()

find_package(nlohmann_json CONFIG QUIET)

if(NCNN_LLM_FETCH_DEPS AND (NOT TARGET ncnn OR NOT TARGET nlohmann_json::nlohmann_json))
    include(FetchContent)

    if(NOT TARGET ncnn)
        set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
        set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
        set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(NCNN_VULKAN OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            ncnn
            GIT_REPOSITORY https://github.com/Tencent/ncnn.git
            GIT_TAG master
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(ncnn)
    endif()

    if(NOT TARGET nlohmann_json::nlohmann_json)
        FetchContent_Declare(
            nlohmann_json
            URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
            URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
        )
        FetchContent_MakeAvailable(nlohmann_json)
    endif()
endif()

if(NOT TARGET ncnn)
    message(FATAL_ERROR
        "ncnn was not found. Set NCNN_SOURCE_DIR, install ncnn's CMake package, "
        "or configure with -DNCNN_LLM_FETCH_DEPS=ON.")
endif()

if(NOT TARGET nlohmann_json::nlohmann_json)
    message(FATAL_ERROR
        "nlohmann_json was not found. Install its CMake package or configure "
        "with -DNCNN_LLM_FETCH_DEPS=ON.")
endif()

add_library(ncnn_tokenizer STATIC
    src/utils/tokenizer/bpe_tokenizer.cpp
    src/utils/tokenizer/unigram_tokenizer.cpp
)
target_include_directories(ncnn_tokenizer PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_library(ncnn_llm_core STATIC
    src/ncnn_text_runtime.cpp
    src/sampling.cpp
    src/utils/gdr.cpp
    src/utils/image_utils.cpp
    src/utils/prompt.cpp
    src/utils/rope_embed.cpp
    src/utils/vision_rope.cpp
)
target_include_directories(ncnn_llm_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(ncnn_llm_core
    PUBLIC
        ncnn
        ncnn_tokenizer
        nlohmann_json::nlohmann_json
)

if(NCNN_LLM_BUILD_OCR)
    add_library(ncnn_llm_ocr STATIC src/ncnn_llm_ocr.cpp)
    target_include_directories(ncnn_llm_ocr PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(ncnn_llm_ocr PUBLIC ncnn_llm_core)

    add_executable(hunyuan_ocr_ncnn examples/ocr_main.cpp)
    target_include_directories(hunyuan_ocr_ncnn PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/examples")
    target_link_libraries(hunyuan_ocr_ncnn PRIVATE ncnn_llm_ocr)
    if(WIN32)
        target_link_libraries(hunyuan_ocr_ncnn PRIVATE shell32)
    endif()
endif()

if(NCNN_LLM_BUILD_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_executable(ncnn_llm_tests tests/test_llm.cpp src/ncnn_llm_gpt.cpp)
        target_include_directories(ncnn_llm_tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests"
        )
        target_link_libraries(ncnn_llm_tests PRIVATE ncnn_llm_core)
        add_test(NAME ncnn_llm_tests COMMAND ncnn_llm_tests)
        set_tests_properties(ncnn_llm_tests PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )
    endif()
endif()
