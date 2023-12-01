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

#include "bench.h"

#include "net.h"
#include "cpu.h" //ncnn


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


    // benchmark("models/nanodet_m.param", ncnn::Mat(320, 320, 3), opt);
    // benchmark(("models/nanodet_m.param").c_str(), ncnn::Mat(320, 320, 3), opt);

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