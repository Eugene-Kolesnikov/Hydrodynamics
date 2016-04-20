#include <cuda.h>
#include "cell.h"
#include "LogSystem/FileLogger.hpp"
#include "cu_gpuProperties.h"

extern "C" void cu_AllocateHostPinnedMemory(void** ptr, int size, void* Log)
{
    HANDLE_CUERROR( cudaHostAlloc(ptr, size * sizeof(Cell), cudaHostAllocDefault) );
}

extern "C" void cu_FreeHostPinnedMemory(Cell* ptr, void* Log)
{
    HANDLE_CUERROR(cudaFreeHost(ptr));
}

extern "C" void cu_AllocateFieldMemory(void* prop, int size)
{
    cu_gpuProperties* gpu = (cu_gpuProperties*) prop;
    logging::FileLogger* Log = gpu->Log;
    HANDLE_CUERROR( cudaMalloc( (void**)&gpu->m_Field, size * sizeof(Cell) ) );
    gpu->m_Field_size = size;
}

extern "C" void cu_loadFieldData(void* prop, Cell* host, int size, int type)
{ // type = { cu_loadFromDeviceToHost, cu_loadFromHostToDevice }
    cu_gpuProperties* gpu = (cu_gpuProperties*) prop;
    logging::FileLogger* Log = gpu->Log;
    if(type == cu_loadFromDeviceToHost) {
        HANDLE_CUERROR( cudaMemcpy( host, gpu->m_Field, size * sizeof(Cell), cudaMemcpyDeviceToHost ) );
    } else if(type == cu_loadFromHostToDevice) {
        HANDLE_CUERROR( cudaMemcpy( gpu->m_Field, host, size * sizeof(Cell), cudaMemcpyHostToDevice ) );
    } else {
        *Log << _WARNING_ << "Wrong 'type' in 'cu_loadFieldData'. Nothing will be done.";
    }
}

extern "C" void cu_loadHaloData(void* prop, Cell* host, int size, int type)
{ // type = { cu_loadFromDeviceToHost, cu_loadFromHostToDevice }
    cu_gpuProperties* gpu = (cu_gpuProperties*) prop;
    logging::FileLogger* Log = gpu->Log;
    if(type == cu_loadFromDeviceToHost) {
        HANDLE_CUERROR( cudaMemcpy( host, gpu->m_Field + 1, size * sizeof(Cell), cudaMemcpyDeviceToHost ) );
        HANDLE_CUERROR( cudaMemcpy( host, gpu->m_Field + gpu->m_Field_size - size - 1, size * sizeof(Cell), cudaMemcpyDeviceToHost ) );
    } else if(type == cu_loadFromHostToDevice) {
        HANDLE_CUERROR( cudaMemcpy( gpu->m_Field + 1, host, size * sizeof(Cell), cudaMemcpyHostToDevice ) );
        HANDLE_CUERROR( cudaMemcpy( gpu->m_Field + gpu->m_Field_size - size - 1, host, size * sizeof(Cell), cudaMemcpyHostToDevice ) );
    } else {
        *Log << _WARNING_ << "Wrong 'type' in 'cu_loadHaloData'. Nothing will be done.";
    }
}

extern "C" void cu_loadBorderData(void* prop, Cell* host, int size, int type)
{ // type = { cu_loadFromDeviceToHost, cu_loadFromHostToDevice }
    cu_gpuProperties* gpu = (cu_gpuProperties*) prop;
    logging::FileLogger* Log = gpu->Log;
    if(type == cu_loadFromDeviceToHost) {
        HANDLE_CUERROR( cudaMemcpy( host, (gpu->m_Field + (size + 2) + 1), size * sizeof(Cell), cudaMemcpyDeviceToHost ) );
        HANDLE_CUERROR( cudaMemcpy( host + size,
            gpu->m_Field + gpu->m_Field_size - 2*(size + 2) + 1, size * sizeof(Cell), cudaMemcpyDeviceToHost ) );
    } else if(type == cu_loadFromHostToDevice) {
        HANDLE_CUERROR( cudaMemcpy( gpu->m_Field + size + 3, host, size * sizeof(Cell), cudaMemcpyHostToDevice ) );
        HANDLE_CUERROR( cudaMemcpy( gpu->m_Field + gpu->m_Field_size - 2*(size + 2) + 1,
            host + size, size * sizeof(Cell), cudaMemcpyHostToDevice ) );
    } else {
        *Log << _WARNING_ << "Wrong 'type' in 'cu_loadHaloData'. Nothing will be done.";
    }
}
