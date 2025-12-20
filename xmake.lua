add_rules("plugin.compile_commands.autoupdate", {outputdir = ".vscode"})
add_rules("mode.debug", "mode.release")

set_encodings("utf-8")

set_languages("c++20", "c11")

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

add_requires("opencv")
add_requires("nlohmann_json")
add_requires("cpp-httplib", {configs = {ssl = false}})

add_includedirs("src/")

target("ncnn_tokenizer")
    set_kind("static")
    add_files("src/utils/tokenizer/*.cpp")

target("ncnn_llm")
    set_kind("static")
    add_files("src/*.cpp")
    add_files("src/utils/*.cpp")
    add_deps("ncnn_tokenizer")
    add_packages("ncnn", "opencv", "nlohmann_json")

function add_example(repo)
    target(repo)
        set_kind("binary")
        add_includedirs("examples/")
        add_files("examples/" .. repo .. ".cpp")
        add_deps("ncnn_llm")
        add_packages("ncnn", "opencv", "nlohmann_json")

        set_rundir("$(projectdir)/")
end

add_example("nllb_main")
add_example("minicpm4_main")
add_example("qwen3_main")
add_example("bytelevelbpe_main")
add_example("qwen2.5_vl_main")

target("qwen3_openai_api")
    set_kind("binary")
    add_includedirs("examples/")
    add_files("examples/qwen3_openai_api.cpp")
    add_deps("ncnn_llm")
    add_packages("ncnn", "opencv", "nlohmann_json", "cpp-httplib")

    set_rundir("$(projectdir)/")

target("llm_ncnn_run")
    set_kind("binary")
    add_includedirs("examples/")
    add_files("examples/llm_ncnn_run/*.cpp")
    add_files("examples/common/*.cpp")
    add_deps("ncnn_llm")
    add_packages("ncnn", "opencv", "nlohmann_json", "cpp-httplib")

    set_rundir("$(projectdir)/")

target("benchllm")
    set_kind("binary")
    add_files("benchmark/benchllm.cpp")

    add_deps("ncnn_llm")
    add_packages("ncnn", "opencv")

    set_rundir("$(projectdir)/assets/minicpm4_0.5b/")
