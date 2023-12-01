#include <coreinit/thread.h>
#include <coreinit/time.h>
#include <coreinit/systeminfo.h>
#include <nn/ac.h>

#include <whb/proc.h>
#include <whb/log.h>
#include <whb/log_console.h>

#include <thread>

#include "net.h"
#include "cpu.h" //ncnn


int main(int argc, char **argv)
{
    // nn::ac::ConfigIdNum configId;

    // nn::ac::Initialize();
    // nn::ac::GetStartupId(&configId);
    // nn::ac::Connect(configId);

    WHBProcInit();
    WHBLogConsoleInit();

    // Print the opening msg
    WHBLogPrintf("Hello 27ovo!\n\n");

    WHBLogPrintf("                     CPU count:       %d", ncnn::get_cpu_count());
    WHBLogPrintf("                     CPU count(BIG):  %d", ncnn::get_big_cpu_count());
    WHBLogPrintf("                     CPU count(SML):  %d", ncnn::get_little_cpu_count());
    WHBLogPrintf("                     CPU powersave:   %d", ncnn::get_cpu_powersave());


    WHBLogPrintf("");
    WHBLogPrintf("");
    WHBLogPrintf("");
    WHBLogPrintf("Hello, NCNN bench demo!");
    WHBLogPrintf("Press HOME to exit");

    //     std::cout << "\x1b[10;16HCPU count: " << ncnn::get_cpu_count() << std::endl;
    // std::cout << "\x1b[11;16HCPU count(BIG): " << ncnn::get_big_cpu_count() << std::endl;
    // std::cout << "\x1b[12;16HCPU count(SML): " << ncnn::get_little_cpu_count() << std::endl;
    // std::cout << "\x1b[13;16HCPU powersave: " << ncnn::get_cpu_powersave() << std::endl;

    // std::cout << "\x1b[16;16HHello, NCNN bench demo!" << std::endl;
    // std::cout << "\x1b[17;16HPress START to exit" << std::endl;


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