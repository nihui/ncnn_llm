add_rules("plugin.compile_commands.autoupdate", {outputdir = ".vscode"})
add_rules("mode.debug", "mode.release")

set_encodings("utf-8")

set_languages("c++20", "c11")

local with_opencv = not is_plat("wasm")
local with_libcurl = not is_plat("wasm")

if with_opencv then
    add_defines("NCNN_LLM_WITH_OPENCV=1")
else
    add_defines("NCNN_LLM_WITH_OPENCV=0")
end

if is_plat("wasm") then
    add_requires("emscripten")
    set_toolchains("emcc@emscripten")
end

if is_plat("windows") then
    add_defines("NOMINMAX")
end

if is_plat("windows", "mingw") then
    add_syslinks("user32", "gdi32")
end

add_requires("ncnn master", {
    configs = {
        simple_vulkan = true,
        with_examples = false,
        with_tests = false,
        with_tools = false
    }
})

if with_opencv then
    add_requires("opencv")
end
add_requires("nlohmann_json")
add_requires("cpp-httplib", {configs = {ssl = false}})
if with_libcurl then
    add_requires("libcurl")
end

add_includedirs("src/")

target("ncnn_tokenizer")
    set_kind("static")
    add_files("src/utils/tokenizer/*.cpp")

target("ncnn_llm")
    set_kind("static")
    add_files("src/*.cpp")
    add_files("src/utils/*.cpp")
    add_deps("ncnn_tokenizer")
    add_packages("ncnn", "nlohmann_json")
    if with_opencv then
        add_packages("opencv")
    end

function add_example(repo)
    target(repo)
        set_kind("binary")
        add_includedirs("examples/")
        add_files("examples/" .. repo .. ".cpp")
        add_deps("ncnn_llm")
        add_packages("ncnn", "nlohmann_json")
        if with_opencv then
            add_packages("opencv")
        end

        set_rundir("$(projectdir)/")
end

add_example("nllb_main")
add_example("minicpm4_main")
add_example("qwen3_main")
add_example("bytelevelbpe_main")
if with_opencv then
    add_example("qwen2.5_vl_main")
end

target("qwen3_openai_api")
    set_kind("binary")
    add_includedirs("examples/")
    add_files("examples/qwen3_openai_api.cpp")
    add_deps("ncnn_llm")
    add_packages("ncnn", "nlohmann_json", "cpp-httplib")
    if with_opencv then
        add_packages("opencv")
    end

    set_rundir("$(projectdir)/")

if with_libcurl then
    target("llm_ncnn_run")
        set_kind("binary")
        add_includedirs("examples/")
        add_files("examples/llm_ncnn_run/*.cpp")
        add_deps("ncnn_llm")
        add_packages("ncnn", "nlohmann_json", "cpp-httplib", "libcurl")
        if with_opencv then
            add_packages("opencv")
        end

        set_rundir("$(projectdir)/")
end

target("benchllm")
    set_kind("binary")
    add_files("benchmark/benchllm.cpp")

    add_deps("ncnn_llm")
    add_packages("ncnn")
    if with_opencv then
        add_packages("opencv")
    end

    set_rundir("$(projectdir)/assets/minicpm4_0.5b/")
