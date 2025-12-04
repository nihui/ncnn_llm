add_rules("plugin.compile_commands.autoupdate", {outputdir = ".vscode"})
add_rules("mode.debug", "mode.release")

set_encodings("utf-8")

set_languages("c++20", "c11")

if is_plat("windows") then
    add_defines("NOMINMAX")
end

add_requires("ncnn master", {
    configs = {
        vulkan = true,
    }
})

add_requires("opencv-mobile")
add_requires("nlohmann_json")

add_includedirs("src/")

target("ncnn_tokenizer")
    set_kind("static")
    add_files("src/utils/tokenizer/*.cpp")

target("ncnn_llm")
    set_kind("static")
    add_files("src/*.cpp")
    add_files("src/utils/*.cpp")
    add_deps("ncnn_tokenizer")
    add_packages("ncnn", "opencv-mobile", "nlohmann_json")

function add_example(repo)
    target(repo)
        set_kind("binary")
        add_includedirs("examples/")
        add_files("examples/" .. repo .. ".cpp")
        add_deps("ncnn_llm")
        add_packages("ncnn", "opencv-mobile", "nlohmann_json")

        set_rundir("$(projectdir)/")
end

add_example("nllb_main")
add_example("minicpm4_main")
add_example("qwen3_main")
add_example("bytelevelbpe_main")

target("benchllm")
    set_kind("binary")
    add_files("benchmark/benchllm.cpp")

    add_deps("ncnn_llm")
    add_packages("ncnn", "opencv-mobile")

    set_rundir("$(projectdir)/assets/minicpm4_0.5b/")