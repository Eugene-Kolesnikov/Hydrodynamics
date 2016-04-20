#include <cuda.h>
#include "cell.h"
#include "LogSystem/FileLogger.hpp"

#define HANDLE_CUERROR(call) {										             \
    cudaError err = call;												         \
    if(err != cudaSuccess) {											         \
        *((logging::FileLogger*)Log) << _ERROR_                                  \
                       << (std::string("CUDA error in file '") +                 \
                           std::string(__FILE__) +  std::string("' in line ") +  \
                           std::to_string(__LINE__) + std::string(": ") +        \
                           std::string(cudaGetErrorString(err))).c_str();        \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",	         \
            __FILE__, __LINE__, cudaGetErrorString(err));				         \
        exit(1);														         \
    }																	         \
} while (0)

extern "C" void cu_AllocateGpuMemory(void** ptr, int size, void* Log)
{
    HANDLE_CUERROR( cudaMalloc( ptr, size ) );
}

extern "C" void cu_AllocateHostPinnedMemory(void** ptr, int size, void* Log)
{
    HANDLE_CUERROR( cudaHostAlloc(ptr, size, cudaHostAllocDefault) );
}

extern "C" void cu_FreeGpuMemory(Cell* ptr, void* Log)
{
    HANDLE_CUERROR(cudaFree(ptr));
}

extern "C" void cu_FreeHostPinnedMemory(Cell* ptr, void* Log)
{
    HANDLE_CUERROR(cudaFreeHost(ptr));
}

extern "C" void cu_loadDataToGpu(Cell* dev, Cell* host, int size, void* Log)
{
    HANDLE_CUERROR( cudaMemcpy( dev, host, size, cudaMemcpyHostToDevice ) );
}

extern "C" void cu_loadDataToHost(Cell* host, Cell* dev, int size, void* Log)
{
    HANDLE_CUERROR( cudaMemcpy( host, dev, size, cudaMemcpyDeviceToHost ) );
}
