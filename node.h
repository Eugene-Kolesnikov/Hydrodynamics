#ifndef NODE_H
#define NODE_H

#include <mpi.h>
#include "cell.h"
extern MPI_Datatype MPI_CellType;

// how long program will work
#define TOTAL_TIME 1.0e-3
// time step
#define TAU 1.0e-3

class Node
{
public:
    Node(const int rank, const int size, const int Nx, const int Ny):
        m_rank(rank), m_size(size), m_Nx(Nx), m_Ny(Ny),
        m_bNx(Nx + 2), m_bNy(Ny + 2) {}
    ~Node(){}

    virtual void runNode() = 0;

protected:
    int m_rank; // current rank of a process
    int m_size; // number of working processes
    int m_Nx;
    int m_Ny;
    int m_bNx;
    int m_bNy;
};

#endif // NODE_H
