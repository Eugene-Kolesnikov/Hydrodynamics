#include "processNode.h"
#include <mpi.h>

ProcessNode::ProcessNode(const int rank, const int size, const int Nx, const int Ny):
    Node::Node(rank, size, Nx, Ny),
    Log(rank, createLogFilename(rank))
{
    int numCompNodes = m_size - 1;
    int numColumns = m_Nx;
    int intNumColumns = numColumns / numCompNodes;
    int intNumPoints = (intNumColumns + 2) * m_bNy; // (adding 2 columns of halo points)
    m_columns = (intNumColumns + 2);
    m_Field = new Cell[m_columns*m_bNy];
    Log << (std::string("Created array of ") + std::to_string(m_columns*m_bNy) + std::string(" elements.")).c_str();
}

ProcessNode::~ProcessNode()
{
    delete m_Field;
}

void ProcessNode::runNode()
{
    initBlock();
    updateBorders(); //cpu version
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
    int intNumPoints = (intNumColumns + 2) * m_bNy;
    Cell* rcv_address = m_Field;
    Log << (std::string("Try to recieve ") + std::to_string(intNumPoints) + std::string(" amount of data from server node.")).c_str();
    MPI_Recv(rcv_address, intNumPoints, MPI_CellType, server_process, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
    Log << "Recieved data from server.";
    MPI_Barrier(MPI_COMM_WORLD);
    Log << "Performed barrier synchronization.";
}

void ProcessNode::sendBlockToServer()
{
    int server_process = m_size - 1;
    int numColumns = m_Nx;
    int numCompNodes = m_size - 1;
    int intNumColumns = numColumns / numCompNodes;
    int num_halo_points = m_bNy;
    // two extreme nodes share less data
    int sendSize = (m_columns - 1) * m_bNy - (m_rank != 0 && m_rank != m_size-2) * m_bNy ;

    Log << (std::string("Try to send ") + std::to_string(sendSize) + std::string(" amount of data to server node.")).c_str();
    Cell* snd_address = m_Field + (m_rank != 0) * num_halo_points; // shift all adresses except the first one
    MPI_Send(snd_address, sendSize, MPI_CellType, server_process, 0, MPI_COMM_WORLD );
    MPI_Barrier(MPI_COMM_WORLD);
    Log << "Performed barrier synchronization.";
}

void ProcessNode::updateBorders()
{
    for (int yIndex = 0; yIndex < m_bNy; ++yIndex)
    {
        for (int xIndex = 0; xIndex < m_columns; ++xIndex)
        {
            if(m_rank == 0) {
                if(xIndex == 0) { //left vertical line
                    if(yIndex == 0 || yIndex == (m_bNy-1)) {
                        m_Field[xIndex * m_bNy + yIndex].r =
                        m_Field[xIndex * m_bNy + yIndex].u =
                        m_Field[xIndex * m_bNy + yIndex].v =
                        m_Field[xIndex * m_bNy + yIndex].e = 0.0;
                    } else {
                        m_Field[xIndex * m_bNy + yIndex].r = m_Field[(xIndex+1) * m_bNy + yIndex].r;
                        m_Field[xIndex * m_bNy + yIndex].u = -m_Field[(xIndex+1) * m_bNy + yIndex].u;
                        m_Field[xIndex * m_bNy + yIndex].v = m_Field[(xIndex+1) * m_bNy + yIndex].v;
                        m_Field[xIndex * m_bNy + yIndex].e = m_Field[(xIndex+1) * m_bNy + yIndex].e;
                    }
                } else {
                    if(yIndex == 0) { // lower horizontal line
                        m_Field[xIndex * m_bNy].r = m_Field[xIndex * m_bNy + 1].r;
                        m_Field[xIndex * m_bNy].u = m_Field[xIndex * m_bNy + 1].u;
                        m_Field[xIndex * m_bNy].v = -m_Field[xIndex * m_bNy + 1].v;
                        m_Field[xIndex * m_bNy].e = m_Field[xIndex * m_bNy + 1].e;
                    } else if(yIndex == (m_bNy-1)) { // upper horizontal line
                        m_Field[xIndex * m_bNy + yIndex].r = m_Field[xIndex * m_bNy + yIndex - 1].r;
                        m_Field[xIndex * m_bNy + yIndex].u = m_Field[xIndex * m_bNy + yIndex - 1].u;
                        m_Field[xIndex * m_bNy + yIndex].v = -m_Field[xIndex * m_bNy + yIndex - 1].v;
                        m_Field[xIndex * m_bNy + yIndex].e = m_Field[xIndex * m_bNy + yIndex - 1].e;
                    }
                }
            } else if(m_rank == (m_size - 2)) {
                if(xIndex == (m_columns-1)) { //right vertical line
                    if(yIndex == 0 || yIndex == (m_bNy-1)) {
                        m_Field[xIndex * m_bNy + yIndex].r =
                        m_Field[xIndex * m_bNy + yIndex].u =
                        m_Field[xIndex * m_bNy + yIndex].v =
                        m_Field[xIndex * m_bNy + yIndex].e = 0.0;
                    } else {
                        m_Field[xIndex * m_bNy + yIndex].r = m_Field[(xIndex-1) * m_bNy + yIndex].r;
                        m_Field[xIndex * m_bNy + yIndex].u = -m_Field[(xIndex-1) * m_bNy + yIndex].u;
                        m_Field[xIndex * m_bNy + yIndex].v = m_Field[(xIndex-1) * m_bNy + yIndex].v;
                        m_Field[xIndex * m_bNy + yIndex].e = m_Field[(xIndex-1) * m_bNy + yIndex].e;
                    }
                } else {
                    if(yIndex == 0) { // lower horizontal line
                        m_Field[xIndex * m_bNy].r = m_Field[xIndex * m_bNy + 1].r;
                        m_Field[xIndex * m_bNy].u = m_Field[xIndex * m_bNy + 1].u;
                        m_Field[xIndex * m_bNy].v = -m_Field[xIndex * m_bNy + 1].v;
                        m_Field[xIndex * m_bNy].e = m_Field[xIndex * m_bNy + 1].e;
                    } else if(yIndex == (m_bNy-1)) { // upper horizontal line
                        m_Field[xIndex * m_bNy + yIndex].r = m_Field[xIndex * m_bNy + yIndex - 1].r;
                        m_Field[xIndex * m_bNy + yIndex].u = m_Field[xIndex * m_bNy + yIndex - 1].u;
                        m_Field[xIndex * m_bNy + yIndex].v = -m_Field[xIndex * m_bNy + yIndex - 1].v;
                        m_Field[xIndex * m_bNy + yIndex].e = m_Field[xIndex * m_bNy + yIndex - 1].e;
                    }
                }
            } else {
                if(yIndex == 0) { // lower horizontal line
                    m_Field[xIndex * m_bNy].r = m_Field[xIndex * m_bNy + 1].r;
                    m_Field[xIndex * m_bNy].u = m_Field[xIndex * m_bNy + 1].u;
                    m_Field[xIndex * m_bNy].v = -m_Field[xIndex * m_bNy + 1].v;
                    m_Field[xIndex * m_bNy].e = m_Field[xIndex * m_bNy + 1].e;
                } else if(yIndex == (m_bNy-1)) { // upper horizontal line
                    m_Field[xIndex * m_bNy + yIndex].r = m_Field[xIndex * m_bNy + yIndex - 1].r;
                    m_Field[xIndex * m_bNy + yIndex].u = m_Field[xIndex * m_bNy + yIndex - 1].u;
                    m_Field[xIndex * m_bNy + yIndex].v = -m_Field[xIndex * m_bNy + yIndex - 1].v;
                    m_Field[xIndex * m_bNy + yIndex].e = m_Field[xIndex * m_bNy + yIndex - 1].e;
                }
            }
        }
    }
}
