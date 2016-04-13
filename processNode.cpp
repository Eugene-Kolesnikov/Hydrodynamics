#include "processNode.h"
#include <mpi.h>

ProcessNode::ProcessNode(const int rank, const int size, const int Nx, const int Ny):
    Node::Node(rank, size, Nx, Ny)
{

}

ProcessNode::~ProcessNode()
{

}

void ProcessNode::runNode()
{

}

void ProcessNode::initBlock()
{

}
