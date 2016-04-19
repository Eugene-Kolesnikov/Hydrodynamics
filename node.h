#ifndef NODE_H
#define NODE_H

#include <mpi.h>
#include "cell.h"
extern MPI_Datatype MPI_CellType;

class Node
{
public:
    Node(const int rank, const int size, const int Nx, const int Ny):
        m_rank(rank), m_size(size), m_Nx(Nx), m_Ny(Ny) {}
    ~Node(){}

    virtual void runNode() = 0;

protected:
    int m_rank; // current rank of a process
    int m_size; // number of working processes
    int m_Nx;
    int m_Ny;
};

#endif // NODE_H
