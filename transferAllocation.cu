#include <cuda.h>
#include "cell.h"
#include "LogSystem/FileLogger.hpp"
#include "cu_gpuProperties.h"

extern "C" void cu_AllocateHostPinnedMemory(void** ptr, int size, void* Log)
{
    HANDLE_CUERROR( cudaHostAlloc(ptr, size, cudaHostAllocDefault) );
}

extern "C" void cu_FreeHostPinnedMemory(Cell* ptr, void* Log)
{
    HANDLE_CUERROR(cudaFreeHost(ptr));
}

extern "C" void cu_AllocateFieldMemory(void* prop, int size)
{
    cu_gpuProperties* gpu = (cu_gpuProperties*) prop;
    logging::FileLogger* Log = gpu->Log;
    HANDLE_CUERROR( cudaMalloc( (void**)&gpu->m_Field, size ) );
}

extern "C" void cu_AllocateHaloMemory(void* prop, int size)
{
    cu_gpuProperties* gpu = (cu_gpuProperties*) prop;
    logging::FileLogger* Log = gpu->Log;
    HANDLE_CUERROR( cudaMalloc( (void**)&gpu->m_halo, size ) );
}

extern "C" void cu_loadFieldData(void* prop, Cell* host, int size, int type)
{ // type = { cu_loadFromDeviceToHost, cu_loadFromHostToDevice }
    cu_gpuProperties* gpu = (cu_gpuProperties*) prop;
    logging::FileLogger* Log = gpu->Log;
    if(type == cu_loadFromDeviceToHost) {
        HANDLE_CUERROR( cudaMemcpy( host, gpu->m_Field, size, cudaMemcpyDeviceToHost ) );
    } else if(type == cu_loadFromHostToDevice) {
        HANDLE_CUERROR( cudaMemcpy( gpu->m_Field, host, size, cudaMemcpyHostToDevice ) );
    } else {
        *Log << _WARNING_ << "Wrong 'type' in 'cu_loadFieldData'. Nothing will be done.";
    }
}

extern "C" void cu_loadHaloData(void* prop, Cell* host, int size, int type)
{ // type = { cu_loadFromDeviceToHost, cu_loadFromHostToDevice }
    cu_gpuProperties* gpu = (cu_gpuProperties*) prop;
    logging::FileLogger* Log = gpu->Log;
    if(type == cu_loadFromDeviceToHost) {
        HANDLE_CUERROR( cudaMemcpy( host, gpu->m_halo, size, cudaMemcpyDeviceToHost ) );
    } else if(type == cu_loadFromHostToDevice) {
        HANDLE_CUERROR( cudaMemcpy( gpu->m_halo, host, size, cudaMemcpyHostToDevice ) );
    } else {
        *Log << _WARNING_ << "Wrong 'type' in 'cu_loadHaloData'. Nothing will be done.";
    }
}
