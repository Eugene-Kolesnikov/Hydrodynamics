#ifndef PROCESSNODE_H
#define PROCESSNODE_H

#include "LogSystem/FileLogger.hpp"
#include "node.h"

class ProcessNode : public Node
{
public:
    ProcessNode(const int rank, const int size, const int Nx, const int Ny);
    ~ProcessNode();

    void runNode();

private:
    void initBlock();

private:
    logging::FileLogger Log;
};

#endif // PROCESSNODE_H
