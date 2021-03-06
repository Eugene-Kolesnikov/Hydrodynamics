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
    *Log << "Successfully updated border elements and synchronized correctly.";
}

extern "C" void cu_computeBorderElements(void* prop)
{
    cu_gpuProperties* gpu = (cu_gpuProperties*) prop;
    logging::FileLogger* Log = gpu->Log;
    Cell* borders = gpu->m_borders;
    Cell* field = gpu->m_Field;
    int Nx = gpu->m_Field_size / gpu->m_bNy - 2; // number of horizontal elements
    int Ny = gpu->m_bNy - 2; // number of vertical elements
    int verticalThreads = 32;
    int verticalBlocks = floor((Ny - 1.0) / (verticalThreads - 2)) + 1;
    cu_computeElements <<< verticalBlocks,
                           dim3(verticalThreads,6,1),
                           verticalThreads * 6 * sizeof(Cell),
                           gpu->streamHaloBorder >>> (borders, field, Nx , Ny, gpu->m_Field_size, Nx * (gpu->m_totalRanks-1), _BORDERS_);
    cudaStreamSynchronize(gpu->streamHaloBorder);
    *Log << "Successfully computed border elements and synchronized correctly.";
}

extern "C" void cu_computeInternalElements(void* prop)
{
    cu_gpuProperties* gpu = (cu_gpuProperties*) prop;
    logging::FileLogger* Log = gpu->Log;
    Cell* borders = gpu->m_borders;
    Cell* field = gpu->m_Field;
    int Nx = gpu->m_Field_size / gpu->m_bNy - 2; // number of horizontal elements
    int Ny = gpu->m_bNy - 2; // number of vertical elements
    int numX = Nx - 2; // number of horizontal elements without halo points and borders
    int numY = Ny; // number of vertical elements without halo points
    int horizontalThreads = 32;
    int horizontalBlocks = floor((numX - 1.0) / (horizontalThreads - 2)) + 1;
    int verticalThreads = 32;
    int verticalBlocks = floor((numY - 1.0) / (verticalThreads - 2)) + 1;
    cu_computeElements <<< dim3(horizontalBlocks,verticalBlocks,1),
                           dim3(horizontalThreads,verticalThreads,1),
                           verticalThreads * horizontalThreads * sizeof(Cell),
                           gpu->streamInternal >>> (borders, field, Nx, Ny, gpu->m_Field_size, Nx * (gpu->m_totalRanks-1), _INTERNAL_);
    cudaStreamSynchronize(gpu->streamInternal);
    *Log << "Successfully computed iternal elements and synchronized correctly.";
}
