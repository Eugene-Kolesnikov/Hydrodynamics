#ifndef PROCESSNODE_H
#define PROCESSNODE_H

#include "LogSystem/FileLogger.hpp"
#include "node.h"

extern "C" void cu_AllocateHostPinnedMemory(void** ptr, int size, void* Log);
extern "C" void cu_FreeHostPinnedMemory(Cell* ptr, void* Log);
extern "C" void* cu_createGpuProperties(logging::FileLogger* log);
extern "C" void cu_destroyGpuProperties(void* prop);
extern "C" void cu_AllocateFieldMemory(void* prop, int size);
extern "C" void cu_AllocateHaloMemory(void* prop, int size);
// type = { cu_loadFromDeviceToHost, cu_loadFromHostToDevice }
extern "C" void cu_loadFieldData(void* prop, Cell* host, int size, int type);
extern "C" void cu_loadHaloData(void* prop, Cell* host, int size, int type);


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
    void* cu_gpuProp;
};

#endif // PROCESSNODE_H
