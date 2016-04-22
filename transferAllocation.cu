#include <cuda.h>
#include "cell.h"
#include "LogSystem/FileLogger.hpp"
#include "cu_gpuProperties.h"

extern "C" void cu_deviceSynchronize()
{
    cudaDeviceSynchronize();
}

extern "C" void cu_allocateHostPinnedMemory(void** ptr, int size, void* Log)
{
    HANDLE_CUERROR( cudaHostAlloc(ptr, size * sizeof(Cell), cudaHostAllocDefault) );
}

extern "C" void cu_freeHostPinnedMemory(Cell* ptr, void* Log)
{
    HANDLE_CUERROR(cudaFreeHost(ptr));
}

extern "C" void cu_allocateFieldMemory(void* prop, int size)
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
        HANDLE_CUERROR( cudaMemcpyAsync( host, gpu->m_Field,
            size * sizeof(Cell), cudaMemcpyDeviceToHost, gpu->streamInternal ) );
        cudaStreamSynchronize(gpu->streamInternal);
        *Log << (std::string("Stream 'streamInternal' transfered array of ") +
            std::to_string(size) + std::string(" elements from device to host and synchronized correctly.")).c_str();
    } else if(type == cu_loadFromHostToDevice) {
        HANDLE_CUERROR( cudaMemcpyAsync( gpu->m_Field, host,
            size * sizeof(Cell), cudaMemcpyHostToDevice, gpu->streamInternal ) );
        cudaStreamSynchronize(gpu->streamInternal);
        *Log << (std::string("Stream 'streamInternal' transfered array of ") +
            std::to_string(size) + std::string(" elements from host to device and synchronized correctly.")).c_str();
    } else {
        *Log << _WARNING_ << "Wrong 'type' in 'cu_loadFieldData'. Nothing will be done.";
    }
}

extern "C" void cu_loadHaloData(void* prop, Cell* host, int size, int type)
{ // type = { cu_loadFromDeviceToHost, cu_loadFromHostToDevice }
    cu_gpuProperties* gpu = (cu_gpuProperties*) prop;
    logging::FileLogger* Log = gpu->Log;
    if(type == cu_loadFromDeviceToHost) {
        HANDLE_CUERROR( cudaMemcpyAsync( host, gpu->m_Field + 1,
            size * sizeof(Cell), cudaMemcpyDeviceToHost, gpu->streamHaloBorder ) );
        HANDLE_CUERROR( cudaMemcpyAsync( host, gpu->m_Field + gpu->m_Field_size - size - 1,
            size * sizeof(Cell), cudaMemcpyDeviceToHost, gpu->streamHaloBorder ) );
        cudaStreamSynchronize(gpu->streamHaloBorder);
        *Log << (std::string("Stream 'streamHaloBorder' transfered two arrays of ") +
            std::to_string(size) + std::string(" elements from device to host and synchronized correctly.")).c_str();
    } else if(type == cu_loadFromHostToDevice) {
        HANDLE_CUERROR( cudaMemcpyAsync( gpu->m_Field + 1, host,
            size * sizeof(Cell), cudaMemcpyHostToDevice, gpu->streamHaloBorder ) );
        HANDLE_CUERROR( cudaMemcpyAsync( gpu->m_Field + gpu->m_Field_size - size - 1, host,
            size * sizeof(Cell), cudaMemcpyHostToDevice, gpu->streamHaloBorder ) );
        cudaStreamSynchronize(gpu->streamHaloBorder);
        *Log << (std::string("Stream 'streamHaloBorder' transfered two arrays of ") +
            std::to_string(size) + std::string(" elements from host to device and synchronized correctly.")).c_str();
    } else {
        *Log << _WARNING_ << "Wrong 'type' in 'cu_loadHaloData'. Nothing will be done.";
    }
}

extern "C" void cu_loadBorderData(void* prop, Cell* host, int size, int type)
{ // type = { cu_loadFromDeviceToHost, cu_loadFromHostToDevice }
    cu_gpuProperties* gpu = (cu_gpuProperties*) prop;
    logging::FileLogger* Log = gpu->Log;
    if(type == cu_loadFromDeviceToHost) {
        HANDLE_CUERROR( cudaMemcpyAsync( host, (gpu->m_Field + (size + 2) + 1),
        size * sizeof(Cell), cudaMemcpyDeviceToHost, gpu->streamHaloBorder ) );
        HANDLE_CUERROR( cudaMemcpyAsync( host + size,
            gpu->m_Field + gpu->m_Field_size - 2*(size + 2) + 1,
            size * sizeof(Cell), cudaMemcpyDeviceToHost, gpu->streamHaloBorder ) );
        cudaStreamSynchronize(gpu->streamHaloBorder);
        *Log << "Stream 'streamHaloBorder' loaded border data from device to host and synchronized correctly.";
    } else if(type == cu_loadFromHostToDevice) {
        HANDLE_CUERROR( cudaMemcpyAsync( gpu->m_Field + (size + 2) + 1, host,
            size * sizeof(Cell), cudaMemcpyHostToDevice, gpu->streamHaloBorder ) );
        HANDLE_CUERROR( cudaMemcpyAsync( gpu->m_Field + gpu->m_Field_size - 2*(size + 2) + 1, host + size,
            size * sizeof(Cell), cudaMemcpyHostToDevice, gpu->streamHaloBorder ) );
        cudaStreamSynchronize(gpu->streamHaloBorder);
        *Log << "Stream 'streamHaloBorder' loaded halo border from host to device and synchronized correctly.";
    } else {
        *Log << _WARNING_ << "Wrong 'type' in 'cu_loadBorderData'. Nothing will be done.";
    }
}
