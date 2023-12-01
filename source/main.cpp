#include <coreinit/thread.h>
#include <coreinit/time.h>
#include <coreinit/systeminfo.h>
#include <nn/ac.h>

#include <whb/proc.h>
#include <whb/log.h>
#include <whb/log_console.h>

#include <thread>


int main(int argc, char **argv)
{
    // nn::ac::ConfigIdNum configId;

    // nn::ac::Initialize();
    // nn::ac::GetStartupId(&configId);
    // nn::ac::Connect(configId);

    WHBProcInit();
    WHBLogConsoleInit();

    // Print the opening msg
    WHBLogPrintf("Hello 27ovo!");

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