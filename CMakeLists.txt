cmake_minimum_required(VERSION 3.20)

project(ncnn_llm VERSION 0.1.0 LANGUAGES C CXX)

option(NCNN_LLM_BUILD_EXAMPLES "Build command-line examples" ON)
option(NCNN_LLM_BUILD_TESTS "Build model-independent tests" ON)
option(NCNN_LLM_FETCH_DEPS "Fetch ncnn and nlohmann_json when packages are unavailable" OFF)
option(NCNN_LLM_ENABLE_VULKAN "Enable Vulkan in an in-tree or fetched ncnn build" OFF)
set(NCNN_SOURCE_DIR "" CACHE PATH "Path to an ncnn source checkout")
set(NCNN_GIT_TAG "13b6d5318c73be53bb386fa51e8067615d0eb7c1" CACHE STRING
    "Pinned ncnn revision used by FetchContent")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/utf-8)
    add_compile_definitions(NOMINMAX)
endif()

function(ncnn_llm_configure_ncnn)
    set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(NCNN_VULKAN ${NCNN_LLM_ENABLE_VULKAN} CACHE BOOL "" FORCE)
endfunction()

if(NCNN_SOURCE_DIR)
    ncnn_llm_configure_ncnn()
    add_subdirectory("${NCNN_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/ncnn")
else()
    find_package(ncnn CONFIG QUIET)
endif()

find_package(nlohmann_json CONFIG QUIET)
find_package(ZLIB QUIET)

if(NCNN_LLM_FETCH_DEPS AND (NOT TARGET ncnn OR NOT TARGET nlohmann_json::nlohmann_json OR NOT TARGET ZLIB::ZLIB))
    include(FetchContent)

    if(NOT TARGET ncnn)
        ncnn_llm_configure_ncnn()
        FetchContent_Declare(
            ncnn
            GIT_REPOSITORY https://github.com/Tencent/ncnn.git
            GIT_TAG ${NCNN_GIT_TAG}
            GIT_SHALLOW FALSE
        )
        FetchContent_MakeAvailable(ncnn)
    endif()

    if(NOT TARGET nlohmann_json::nlohmann_json)
        FetchContent_Declare(
            nlohmann_json
            URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
            URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(nlohmann_json)
    endif()

    if(NOT TARGET ZLIB::ZLIB)
        set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            zlib
            URL https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
            URL_HASH SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(zlib)
        if(TARGET zlibstatic)
            add_library(ZLIB::ZLIB ALIAS zlibstatic)
        elseif(TARGET zlib)
            add_library(ZLIB::ZLIB ALIAS zlib)
        endif()
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

add_library(ncnn_llm STATIC
    src/ncnn_embedding.cpp
    src/ncnn_llm_asr.cpp
    src/ncnn_llm_gpt.cpp
    src/ncnn_llm_ocr.cpp
    src/ncnn_text_runtime.cpp
    src/nllb_600m.cpp
    src/sampling.cpp
    src/youtu/youtu_app.cpp
    src/youtu/youtu_cli.cpp
    src/youtu/youtu_generation.cpp
    src/youtu/youtu_model_paths.cpp
    src/youtu/youtu_runner_common.cpp
    src/youtu/youtu_tokenizer.cpp
    src/youtu/youtu_vision.cpp
    src/utils/gdr.cpp
    src/utils/image_utils.cpp
    src/utils/prompt.cpp
    src/utils/rope_embed.cpp
    src/utils/vision_rope.cpp
)
target_include_directories(ncnn_llm PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/youtu"
)
if(NCNN_SOURCE_DIR AND EXISTS "${NCNN_SOURCE_DIR}/src/stb_image.h")
    target_include_directories(ncnn_llm PRIVATE "${NCNN_SOURCE_DIR}/src")
else()
    find_path(NCNN_LLM_STB_IMAGE_INCLUDE_DIR stb_image.h)
    if(NOT NCNN_LLM_STB_IMAGE_INCLUDE_DIR)
        message(FATAL_ERROR "stb_image.h is required for Youtu-VL image decoding")
    endif()
    target_include_directories(ncnn_llm PRIVATE "${NCNN_LLM_STB_IMAGE_INCLUDE_DIR}")
endif()

if(NOT TARGET ZLIB::ZLIB)
    message(FATAL_ERROR
        "zlib was not found. Install its CMake package or configure "
        "with -DNCNN_LLM_FETCH_DEPS=ON.")
endif()
target_link_libraries(ncnn_llm
    PUBLIC
        ncnn
        ncnn_tokenizer
        nlohmann_json::nlohmann_json
        ZLIB::ZLIB
)

if(NCNN_LLM_BUILD_EXAMPLES)
    add_executable(llm_ncnn_run
        examples/llm_ncnn_run/cli_runner.cpp
        examples/llm_ncnn_run/json_utils.cpp
        examples/llm_ncnn_run/main.cpp
        examples/llm_ncnn_run/options.cpp
        examples/llm_ncnn_run/tools.cpp
        examples/llm_ncnn_run/util.cpp
    )
    target_include_directories(llm_ncnn_run PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/examples")
    target_link_libraries(llm_ncnn_run PRIVATE ncnn_llm)

    add_executable(ocr_main examples/ocr_main.cpp)
    target_include_directories(ocr_main PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/examples")
    target_link_libraries(ocr_main PRIVATE ncnn_llm)

    add_executable(asr_main examples/asr_main.cpp)
    target_include_directories(asr_main PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/examples")
    target_link_libraries(asr_main PRIVATE ncnn_llm)

    add_executable(youtu_vl_main examples/youtu_vl_main.cpp)
    target_link_libraries(youtu_vl_main PRIVATE ncnn_llm)

    add_executable(nllb_main examples/nllb_main.cpp)
    target_link_libraries(nllb_main PRIVATE ncnn_llm)

    add_executable(embedding_main examples/embedding_main.cpp)
    target_link_libraries(embedding_main PRIVATE ncnn_llm)

    add_executable(clip_main examples/clip_main.cpp)
    target_link_libraries(clip_main PRIVATE ncnn_llm)

    add_executable(bytelevelbpe_main examples/bytelevelbpe_main.cpp)
    target_link_libraries(bytelevelbpe_main PRIVATE ncnn_tokenizer)

    add_executable(unigram_main examples/unigram_main.cpp)
    target_link_libraries(unigram_main PRIVATE ncnn_tokenizer)

    if(WIN32)
        target_link_libraries(llm_ncnn_run PRIVATE shell32)
        target_link_libraries(ocr_main PRIVATE shell32)
        target_link_libraries(asr_main PRIVATE shell32)
        target_link_libraries(youtu_vl_main PRIVATE shell32)
    endif()
endif()

include(GNUInstallDirs)
install(TARGETS ncnn_llm ncnn_tokenizer
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
if(NCNN_LLM_BUILD_EXAMPLES)
    install(TARGETS llm_ncnn_run ocr_main asr_main youtu_vl_main nllb_main embedding_main clip_main
        bytelevelbpe_main unigram_main
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

if(NCNN_LLM_BUILD_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_executable(ncnn_llm_tests tests/test_llm.cpp)
        target_include_directories(ncnn_llm_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
        target_link_libraries(ncnn_llm_tests PRIVATE ncnn_llm)
        add_test(NAME ncnn_llm_tests COMMAND ncnn_llm_tests)
        set_tests_properties(ncnn_llm_tests PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )

        if(NCNN_LLM_BUILD_EXAMPLES)
            add_test(NAME youtu_vl_help COMMAND youtu_vl_main --help)
        endif()

        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            add_test(
                NAME ncnn_llm_tool_tests
                COMMAND "${Python3_EXECUTABLE}" -m unittest discover
                    -s "${CMAKE_CURRENT_SOURCE_DIR}/tests/python" -v
            )
            set_tests_properties(ncnn_llm_tool_tests PROPERTIES
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            )
        endif()
    endif()
endif()
