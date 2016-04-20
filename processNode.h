#ifndef PROCESSNODE_H
#define PROCESSNODE_H

#include "LogSystem/FileLogger.hpp"
#include "node.h"

extern "C" void cu_AllocateGpuMemory(void** ptr, int size, void* Log);
extern "C" void cu_AllocateHostPinnedMemory(void** ptr, int size, void* Log);
extern "C" void cu_FreeGpuMemory(Cell* ptr, void* Log);
extern "C" void cu_FreeHostPinnedMemory(Cell* ptr, void* Log);
extern "C" void cu_loadDataToGpu(Cell* dev, Cell* host, int size, void* Log);
extern "C" void cu_loadDataToHost(Cell* host, Cell* dev, int size, void* Log);

class ProcessNode : public Node
{
public:
    ProcessNode(const int rank, const int size, const int Nx, const int Ny);
    ~ProcessNode();

    void runNode();

private:
    void initBlock();
    void sendBlockToServer();
    void updateBorders();
    void setStopCheckMark();

private:
    Cell* m_Field;
    Cell* m_haloElements;
    int m_columns;
    logging::FileLogger Log;
    double m_time;

private:
    Cell* m_device_Field;
};

#endif // PROCESSNODE_H
