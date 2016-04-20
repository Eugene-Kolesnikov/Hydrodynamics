#include "cu_gpuProperties.h"

cu_gpuProperties::cu_gpuProperties(logging::FileLogger* log):
    Log(log), m_Field_size(0), m_halo_size(0) {}

cu_gpuProperties::~cu_gpuProperties()
{
    if(m_Field_size != 0)
        HANDLE_CUERROR(cudaFree(m_Field));
    if(m_halo_size != 0)
        HANDLE_CUERROR(cudaFree(m_halo));
}

extern "C" void* cu_createGpuProperties(logging::FileLogger* log)
{
    return (void*) new cu_gpuProperties(log);
}

extern "C" void cu_destroyGpuProperties(void* prop)
{
    delete (cu_gpuProperties*) prop;
}
