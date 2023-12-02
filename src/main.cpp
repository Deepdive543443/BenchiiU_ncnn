#include <coreinit/thread.h>
#include <coreinit/time.h>
#include <coreinit/systeminfo.h>
#include <nn/ac.h>

#include <whb/proc.h>
#include <whb/log.h>
#include <whb/log_console.h>

#include <thread>
#include <iostream> 
#include <string>
#include <cfloat>

// #include "bench.h"
#include "datareader.h" 
#include "net.h"
#include "cpu.h" //ncnn


static int g_warmup_loop_count = 8;
static int g_loop_count = 4;
// static bool g_enable_cooling_down = true;

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

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

double get_current_time()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto usec = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    return usec.count() / 1000.0;
}


int main(int argc, char **argv)
{
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 1;
    opt.use_winograd_convolution = true;
    opt.use_sgemm_convolution = true;
    opt.use_int8_inference = true;
    opt.use_vulkan_compute = false;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;
    opt.use_int8_storage = true;
    opt.use_int8_arithmetic = true;
    opt.use_packing_layout = true;
    opt.use_shader_pack8 = false;
    opt.use_image_storage = false;

    WHBProcInit();
    WHBLogConsoleInit();

    // Print the opening msg
    WHBLogPrintf("Hello 27ovo!\n\n");
    WHBLogPrintf("");
    WHBLogPrintf("");

    WHBLogPrintf("                     CPU count:       %d", ncnn::get_cpu_count());
    WHBLogPrintf("                     CPU count(BIG):  %d", ncnn::get_big_cpu_count());
    WHBLogPrintf("                     CPU count(SML):  %d", ncnn::get_little_cpu_count());
    WHBLogPrintf("                     CPU powersave:   %d", ncnn::get_cpu_powersave());


    WHBLogPrintf("");
    WHBLogPrintf("");
    WHBLogPrintf("");
    WHBLogPrintf("Hello, NCNN bench demo!");
    WHBLogPrintf("Press HOME to exit");

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

    ncnn::Net net;
    net.opt = opt;

    // Load weight
    net.load_param("fs:/vol/external01/models/FastestDet.param");
    DataReaderFromEmpty dr;
    net.load_model(dr);

    const std::vector<const char*>& input_names = net.input_names();
    const std::vector<const char*>& output_names = net.output_names();

    ncnn::Mat _in(320, 320, 3);
    
    ncnn::Mat in = _in;
    in.fill(0.01f);

    WHBLogPrintf("Input shape:[H:%d, W:%d, C:%d]\n", in.h, in.w, in.c);
    WHBLogPrintf("Warm up Fastestdet...");
    for (int i = 0; i < g_warmup_loop_count; i++)
    {
        ncnn::Extractor ex = net.create_extractor();
        for (size_t j = 0; j < input_names.size(); ++j)
        {
            ncnn::Mat in = _in;
            ex.input(input_names[j], in);
        }

        for (size_t j = 0; j < output_names.size(); ++j)
        {
            ncnn::Mat out;
            ex.extract(output_names[j], out);
        }

        // WHBLogPrintf("[%2d/%2d]\n",i + 1, g_warmup_loop_count);
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    WHBLogPrintf("Benchmarking...");

    int out_w;
    int out_h;
    int out_c;

    for (int i = 0; i < g_loop_count; i++)
    {
        double start = get_current_time();
        {
            ncnn::Extractor ex = net.create_extractor();
            for (size_t j = 0; j < input_names.size(); ++j)
            {
                ncnn::Mat in = _in;
                ex.input(input_names[j], in);
            }

            for (size_t j = 0; j < output_names.size(); ++j)
            {
                ncnn::Mat out;
                ex.extract(output_names[j], out);
                
                //
                out_w = out.w;
                out_h = out.h;
                out_c = out.c;
            }
        }
        double end = get_current_time();
        double time = end - start;

        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;
        WHBLogPrintf("[%2d/%2d] Time:%7.2f\n",i + 1, g_loop_count, time);
    }

    WHBLogPrintf("Output shape:[H:%d, W:%d, C:%d]\n\n", out_h, out_w, out_c);

    time_avg /= g_loop_count;
    WHBLogPrintf("min = %7.2f  max = %7.2f  avg = %7.2f\n", time_min, time_max, time_avg);

    WHBLogPrintf("Press HOME to exit");

    while(WHBProcIsRunning()) 
    {
        WHBLogConsoleDraw();
        OSSleepTicks(OSMillisecondsToTicks(100));
    }

    WHBLogPrintf("Exiting... ");
    WHBLogConsoleDraw();
    OSSleepTicks(OSMillisecondsToTicks(1000));

    WHBLogConsoleFree();
    WHBProcShutdown();

    // nn::ac::Finalize();
    return 0;
}