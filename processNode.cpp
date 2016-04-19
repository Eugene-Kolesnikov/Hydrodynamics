#include "processNode.h"
#include <mpi.h>

ProcessNode::ProcessNode(const int rank, const int size, const int Nx, const int Ny):
    Node::Node(rank, size, Nx, Ny),
    Log(rank, createLogFilename(rank))
{
    int numCompNodes = m_size - 1;
    int numColumns = m_Nx;
    int intNumColumns = numColumns / numCompNodes;
    int intNumPoints = (intNumColumns + 2) * m_Ny; // (adding 2 columns of halo points)
    m_Field = new Cell[intNumPoints];
}

ProcessNode::~ProcessNode()
{
    delete m_Field;
}

void ProcessNode::runNode()
{
    initBlock();
    sendBlockToServer();
}

void ProcessNode::initBlock()
{
    Log << "Sharing the initialized dense field.";
    MPI_Status status;
    int server_process = m_size - 1;
    int numColumns = m_Nx;
    int numCompNodes = m_size - 1;
    int intNumColumns = numColumns / numCompNodes;
    int intNumPoints = (intNumColumns + 2) * m_Ny;
    int num_halo_points = m_Ny;
    Cell* rcv_address = m_Field + num_halo_points * (0 == m_rank);
    MPI_Recv(rcv_address, intNumPoints, MPI_CellType, server_process, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
    Log << "Recieved data from server.";
    MPI_Barrier(MPI_COMM_WORLD);
    Log << "Perfored barrier synchronization.";

    // FOR DEBUG ---------------------------------------
    //for(int i = 0; i < intNumPoints; ++i)
    //    m_Field[i].r = m_rank;
    // FOR DEBUG ---------------------------------------
}

void ProcessNode::sendBlockToServer()
{
    int server_process = m_size - 1;
    int numColumns = m_Nx;
    int numCompNodes = m_size - 1;
    int intNumColumns = numColumns / numCompNodes;
    int num_halo_points = m_Ny;

    Cell* snd_address = m_Field + num_halo_points;
    MPI_Send(snd_address, intNumColumns * m_Ny, MPI_CellType, server_process, 0, MPI_COMM_WORLD );
    MPI_Barrier(MPI_COMM_WORLD);
}
