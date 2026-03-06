add_rules("plugin.compile_commands.autoupdate", {outputdir = ".vscode"})
add_rules("mode.debug", "mode.release")

set_encodings("utf-8")

set_languages("c++20", "c11")

local with_opencv = not is_plat("wasm")

if with_opencv then
    add_defines("NCNN_LLM_WITH_OPENCV=1")
else
    add_defines("NCNN_LLM_WITH_OPENCV=0")
end

if is_plat("wasm") then
    add_requires("emscripten")
    set_toolchains("emcc@emscripten")
    add_ldflags("-sASSERTIONS=2", "-sDEMANGLE_SUPPORT=1", "-sEXPORTED_RUNTIME_METHODS=['FS']")
end

if is_plat("wasm") and is_arch("wasm64") then
    add_cxflags("-sMEMORY64=1")
    add_ldflags("-sMEMORY64=1", "-sWASM_BIGINT=1")
    add_ldflags("-sINITIAL_MEMORY=1073741824", "-sMAXIMUM_MEMORY=17179869184")
end

if is_plat("windows") then
    add_defines("NOMINMAX")

    add_cxflags("/utf-8")
    add_cxxflags("/utf-8")
end

if is_plat("windows", "mingw") then
    add_syslinks("user32", "gdi32")
end

add_requires("ncnn master", {
    configs = {
        vulkan=true
    }
})

if with_opencv then
    add_requires("opencv")
end
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
    add_packages("ncnn", "nlohmann_json")
    if with_opencv then
        add_packages("opencv")
    end

target("llm_ncnn_run")
    set_kind("binary")
    add_includedirs("examples/")
    add_files("examples/llm_ncnn_run/*.cpp")
    add_deps("ncnn_llm")
    add_packages("ncnn", "nlohmann_json")
    if with_opencv then
        add_packages("opencv")
    end

    set_rundir("$(projectdir)/")

target("benchllm")
    set_kind("binary")
    add_files("benchmark/benchllm.cpp")

    add_deps("ncnn_llm")
    add_packages("ncnn")
    if with_opencv then
        add_packages("opencv")
    end

    set_rundir("$(projectdir)/assets/minicpm4_0.5b/")

target("test_llm")
    set_kind("binary")
    add_includedirs("tests/")
    add_files("tests/test_llm.cpp")
    add_deps("ncnn_llm")
    add_packages("ncnn", "nlohmann_json")
    if with_opencv then
        add_packages("opencv")
    end

    set_rundir("$(projectdir)/")
