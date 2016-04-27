#include "cu_gpuProperties.h"
#include <math.h>

cu_gpuProperties::cu_gpuProperties(logging::FileLogger* log, int bNy, int rank, int totalRanks):
    Log(log), m_Field_size(0), m_bNy(bNy), m_rank(rank), m_totalRanks(totalRanks) {
        cudaStreamCreate(&streamInternal);
        cudaStreamCreate(&streamHaloBorder);

        int numHaloElements = 2 * (m_bNy - 2);
        HANDLE_CUERROR( cudaMalloc( (void**)&m_borders, numHaloElements * sizeof(Cell) ) );
        m_borders_size = numHaloElements;
        *Log << (std::string("Allocated array of ") + std::to_string(numHaloElements) + std::string(" border elements on GPU.")).c_str();
    }

cu_gpuProperties::~cu_gpuProperties()
{
    if(m_Field_size != 0)
        HANDLE_CUERROR(cudaFree(m_Field));
}

extern "C" void* cu_createGpuProperties(logging::FileLogger* log, int bNy, int rank, int totalRanks)
{
    return (void*) new cu_gpuProperties(log, bNy, rank, totalRanks);
}

extern "C" void cu_destroyGpuProperties(void* prop)
{
    delete (cu_gpuProperties*) prop;
}

extern "C" void cu_updateBorders(void* prop)
{
    cu_gpuProperties* gpu = (cu_gpuProperties*) prop;
    logging::FileLogger* Log = gpu->Log;
    int Nx = gpu->m_Field_size / gpu->m_bNy - 2; // number of horizontal border elements
    int Ny = gpu->m_bNy - 2; // number of vertical border elements
    // update horizontal border elements
    int horizontalBlocks = floor((Nx - 1) / 256.0) + 1;
    int horizontalThreads = 256.0;
    updateBordersKernel <<< horizontalBlocks,
                            horizontalThreads,
                            0, gpu->streamHaloBorder >>> (gpu->m_Field, Nx, Ny, 'h', gpu->m_rank, gpu->m_totalRanks);
    // update vertical border elements
    int verticalBlocks = floor((Ny - 1) / 256.0) + 1;
    int verticalThreads = 256.0;
    updateBordersKernel <<< verticalBlocks,
                            verticalThreads,
                            0, gpu->streamHaloBorder >>> (gpu->m_Field, Nx, Ny, 'v', gpu->m_rank, gpu->m_totalRanks);
    cudaStreamSynchronize(gpu->streamHaloBorder);
    *Log << "Successfully updated border elements.";
}

extern "C" void cu_computeBorderElements(void* prop)
{
    cu_gpuProperties* gpu = (cu_gpuProperties*) prop;
    logging::FileLogger* Log = gpu->Log;
    Cell* borders = gpu->m_borders;
    Cell* field = gpu->m_Field;
    int Ny = gpu->m_bNy - 2;
    int verticalBlocks = floor((2*Ny - 1) / 256.0) + 1;
    int verticalThreads = 256.0;
    cu_computeBorders <<< verticalBlocks,
                          verticalThreads,
                          0, gpu->streamHaloBorder >>> (borders, field, Ny, gpu->m_Field_size);
    cudaStreamSynchronize(gpu->streamHaloBorder);
    *Log << "Successfully updated border elements.";
}
