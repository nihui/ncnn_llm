
#include <float.h>
#include <stdio.h>
#include <string.h>

#include <ncnn/net.h>
#include <ncnn/benchmark.h>
#include <ncnn/cpu.h>
#include <ncnn/datareader.h>
#include <ncnn/layer_type.h>

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
    virtual int scan(const char* format, void* p) const
    {
        return 0;
    }
    virtual size_t read(void* buf, size_t size) const
    {
        memset(buf, 0, size);
        return size;
    }
};

static int g_warmup_loop_count = 8;
static int g_loop_count = 4;
static bool g_enable_cooling_down = true;

void benchmark(const char* comment, int hidden_size, int half_embed_dim, int seqlen, const ncnn::Option& opt)
{
    std::string param_path = std::string(comment) + "_decoder.ncnn.param";

    ncnn::Net net;

    net.opt = opt;

    net.load_param(param_path.c_str());

    DataReaderFromEmpty dr;
    net.load_model(dr);

    // resolve kv cache blob indexes
    std::vector<int> kv_cache_indexes;
    std::vector<int> out_kv_cache_indexes;
    {
        for (size_t i = 0; i < net.layers().size(); i++)
        {
            const ncnn::Layer* op = net.layers()[i];
            if (op->typeindex != ncnn::LayerType::SDPA)
                continue;

            const size_t input_count = op->bottoms.size();
            const size_t output_count = op->tops.size();

            if (output_count == 3)
            {
                kv_cache_indexes.push_back(op->bottoms[input_count - 2]);
                kv_cache_indexes.push_back(op->bottoms[input_count - 1]);
                out_kv_cache_indexes.push_back(op->tops[output_count - 2]);
                out_kv_cache_indexes.push_back(op->tops[output_count - 1]);
            }
        }
    }

    if (g_enable_cooling_down)
    {
        // sleep 10 seconds for cooling down SOC  :(
        ncnn::sleep(10 * 1000);
    }

    std::vector<ncnn::Mat> kvcache;

    // prefill
    {
        const int cur_seqlen = seqlen;
        const int past_seqlen = 0;

        ncnn::Mat token_embeds(hidden_size, cur_seqlen);
        ncnn::Mat attention_mask(past_seqlen + cur_seqlen, cur_seqlen);
        ncnn::Mat cos_cache(half_embed_dim, cur_seqlen);
        ncnn::Mat sin_cache(half_embed_dim, cur_seqlen);

        std::vector<ncnn::Mat> out_kvcache;
        ncnn::Mat output_states;

        // warm up
        for (int i = 0; i < g_warmup_loop_count; i++)
        {
            ncnn::Extractor ex = net.create_extractor();
            ex.input("in0", token_embeds);
            ex.input("in1", attention_mask);
            ex.input("in2", cos_cache);
            ex.input("in3", sin_cache);

            // extract updated kv cache
            out_kvcache.resize(out_kv_cache_indexes.size());
            for (size_t i = 0; i < out_kv_cache_indexes.size(); i++)
            {
                ex.extract(out_kv_cache_indexes[i], out_kvcache[i], 1);
            }

            ex.extract("out0", output_states);
        }

        double time_min = DBL_MAX;
        double time_max = -DBL_MAX;
        double time_avg = 0;

        for (int i = 0; i < g_loop_count; i++)
        {
            double start = ncnn::get_current_time();
            {
                ncnn::Extractor ex = net.create_extractor();
                ex.input("in0", token_embeds);
                ex.input("in1", attention_mask);
                ex.input("in2", cos_cache);
                ex.input("in3", sin_cache);

                // extract updated kv cache
                out_kvcache.resize(out_kv_cache_indexes.size());
                for (size_t i = 0; i < out_kv_cache_indexes.size(); i++)
                {
                    ex.extract(out_kv_cache_indexes[i], out_kvcache[i], 1);
                }

                ex.extract("out0", output_states);
            }

            double end = ncnn::get_current_time();

            double time = end - start;

            time_min = std::min(time_min, time);
            time_max = std::max(time_max, time);
            time_avg += time;
        }

        time_avg /= g_loop_count;

        fprintf(stderr, "%20s (prefill)  min = %7.2f  max = %7.2f  avg = %7.2f\n", comment, time_min, time_max, time_avg);

        kvcache = out_kvcache;
    }

    // decode step
    {
        const int cur_seqlen = 1;
        const int past_seqlen = seqlen;

        ncnn::Mat token_embeds(hidden_size, cur_seqlen);
        ncnn::Mat attention_mask(past_seqlen + cur_seqlen, cur_seqlen);
        ncnn::Mat cos_cache(half_embed_dim, cur_seqlen);
        ncnn::Mat sin_cache(half_embed_dim, cur_seqlen);

        std::vector<ncnn::Mat> out_kvcache;
        ncnn::Mat output_states;

        // warm up
        for (int i = 0; i < g_warmup_loop_count; i++)
        {
            ncnn::Extractor ex = net.create_extractor();
            ex.input("in0", token_embeds);
            ex.input("in1", attention_mask);
            ex.input("in2", cos_cache);
            ex.input("in3", sin_cache);

            // pass in kv cache from previous steps
            for (size_t i = 0; i < kv_cache_indexes.size(); i++)
            {
                ex.input(kv_cache_indexes[i], kvcache[i]);
            }

            // extract updated kv cache
            out_kvcache.resize(out_kv_cache_indexes.size());
            for (size_t i = 0; i < out_kv_cache_indexes.size(); i++)
            {
                ex.extract(out_kv_cache_indexes[i], out_kvcache[i], 1);
            }

            ex.extract("out0", output_states);
        }

        double time_min = DBL_MAX;
        double time_max = -DBL_MAX;
        double time_avg = 0;

        for (int i = 0; i < g_loop_count; i++)
        {
            double start = ncnn::get_current_time();
            {
                ncnn::Extractor ex = net.create_extractor();
                ex.input("in0", token_embeds);
                ex.input("in1", attention_mask);
                ex.input("in2", cos_cache);
                ex.input("in3", sin_cache);

                // extract updated kv cache
                out_kvcache.resize(out_kv_cache_indexes.size());
                for (size_t i = 0; i < out_kv_cache_indexes.size(); i++)
                {
                    ex.extract(out_kv_cache_indexes[i], out_kvcache[i], 1);
                }

                ex.extract("out0", output_states);
            }

            double end = ncnn::get_current_time();

            double time = end - start;

            time_min = std::min(time_min, time);
            time_max = std::max(time_max, time);
            time_avg += time;
        }

        time_avg /= g_loop_count;

        fprintf(stderr, "%20s  (decode)  min = %7.2f  max = %7.2f  avg = %7.2f\n", comment, time_min, time_max, time_avg);
    }
}

int main(int argc, char** argv)
{
    int loop_count = 4;
    int num_threads = ncnn::get_physical_big_cpu_count();
    int powersave = 2;
    int gpu_device = -1;
    int cooling_down = 1;
    int seqlen = 233;

    if (argc >= 2)
    {
        loop_count = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        num_threads = atoi(argv[2]);
    }
    if (argc >= 4)
    {
        powersave = atoi(argv[3]);
    }
    if (argc >= 5)
    {
        gpu_device = atoi(argv[4]);
    }
    if (argc >= 6)
    {
        cooling_down = atoi(argv[5]);
    }
    if (argc >= 7)
    {
        seqlen = atoi(argv[6]);
    }

    bool use_vulkan_compute = gpu_device != -1;

    g_enable_cooling_down = cooling_down != 0;

    g_loop_count = loop_count;

    ncnn::set_cpu_powersave(powersave);

    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    // default option
    ncnn::Option opt;
    opt.num_threads = num_threads;
    opt.use_vulkan_compute = use_vulkan_compute;

    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    fprintf(stderr, "gpu_device = %d\n", gpu_device);
    fprintf(stderr, "cooling_down = %d\n", (int)g_enable_cooling_down);
    fprintf(stderr, "seqlen = %d\n", seqlen);

    // run default cases
    benchmark("minicpm4", 1024, 32, seqlen, opt);

    return 0;
}
