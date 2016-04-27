#ifndef TEST_H
#define TEST_H

#include <cuda.h>
#include "LogSystem/FileLogger.hpp"
#include "cell.h"

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

class cu_gpuProperties {
public:
    cu_gpuProperties(logging::FileLogger* log, int bNy, int rank, int totalRanks);
    ~cu_gpuProperties();
public:
    logging::FileLogger* Log;
    int m_rank;
    int m_totalRanks;
public:
    Cell* m_Field;
    int m_Field_size;
    int m_bNy;

    Cell* m_borders;
    int m_borders_size;
public:
    cudaStream_t streamInternal;
    cudaStream_t streamHaloBorder;
};

__global__ void updateBordersKernel(Cell* field, int Nx, int Ny, char type, int rank, int totalRanks);
__global__ void cu_computeBorders(Cell* borders, Cell* field, int Ny, int fieldSize);

#endif
