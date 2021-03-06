#ifndef PROCESSNODE_H
#define PROCESSNODE_H

#include "LogSystem/FileLogger.hpp"
#include "node.h"

extern "C" void cu_deviceSynchronize();
extern "C" void cu_allocateHostPinnedMemory(void** ptr, int size, void* Log);
extern "C" void cu_freeHostPinnedMemory(Cell* ptr, void* Log);
extern "C" void* cu_createGpuProperties(logging::FileLogger* log, int bNy, int rank, int totalRanks);
extern "C" void cu_destroyGpuProperties(void* prop);
extern "C" void cu_allocateFieldMemory(void* prop, int size);
// type = { cu_loadFromDeviceToHost, cu_loadFromHostToDevice }
extern "C" void cu_loadFieldData(void* prop, Cell* host, int size, int type);
extern "C" void cu_loadHaloData(void* prop, Cell* host, int size, int type);
extern "C" void cu_loadBorderData(void* prop, Cell* host, int size, int type);
extern "C" void cu_moveBorderDataToField(void* prop);
extern "C" void cu_updateBorders(void* prop);
extern "C" void cu_computeBorderElements(void* prop);
extern "C" void cu_computeInternalElements(void* prop);

class ProcessNode : public Node
{
public:
    ProcessNode(const int rank, const int size, const int Nx, const int Ny);
    ~ProcessNode();

    void runNode();

private:
    void initBlock();
    void sendBlockToServer();
    void setStopCheckMark();
    void exchangeBorderPoints();

private:
    Cell* m_Field;
    Cell* m_haloElements;
    Cell* m_borderElements;
    int m_columns;
    logging::FileLogger Log;
    double m_time;

private:
    void* cu_gpuProp;
};

#endif // PROCESSNODE_H
