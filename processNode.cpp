#include "processNode.h"
#include <mpi.h>
#include "debug.h"

ProcessNode::ProcessNode(const int rank, const int size, const int Nx, const int Ny):
    Node::Node(rank, size, Nx, Ny),
    Log(rank, createLogFilename(rank))
{
    cu_gpuProp = cu_createGpuProperties(&Log);
    Log << std::string("GPU properties created.").c_str();
    int numCompNodes = m_size - 1;
    int numColumns = m_Nx;
    int intNumColumns = numColumns / numCompNodes;
    int intNumPoints = (intNumColumns + 2) * m_bNy; // (adding 2 columns of halo points)
    int numHaloElements = 2 * m_Ny;
    m_columns = (intNumColumns + 2);
    m_Field = new Cell[intNumPoints];
    Log << (std::string("Allocated array of ") + std::to_string(intNumPoints) + std::string(" Field elements.")).c_str();
    m_borderElements = new Cell[numHaloElements]; // the number of border elements equal to the number of halo elements
    Log << (std::string("Allocated array of ") + std::to_string(numHaloElements) + std::string(" border elements.")).c_str();
    cu_AllocateHostPinnedMemory((void **)&m_haloElements, numHaloElements, &Log);
    Log << (std::string("Allocated array of ") + std::to_string(numHaloElements) + std::string(" elements on Host in pinned memory using cudaHostAlloc.")).c_str();
    cu_AllocateFieldMemory(cu_gpuProp, intNumPoints);
    Log << (std::string("Allocated array of ") + std::to_string(intNumPoints) + std::string(" Field elements on GPU.")).c_str();
}

ProcessNode::~ProcessNode()
{
    delete m_Field;
    cu_FreeHostPinnedMemory(m_haloElements, &Log);
    cu_destroyGpuProperties(cu_gpuProp);
}

void ProcessNode::runNode()
{
    int intNumPoints = m_columns * m_bNy;
    initBlock();
    #ifdef _DEBUG_
        updateBorders(); //cpu version
        writeFieldPart(m_Field, m_columns, m_bNy, Log, "Part of x-velocity field with updated borders");
    #endif
    cu_loadFieldData(cu_gpuProp, m_Field, intNumPoints, cu_loadFromHostToDevice);
    Log << (std::string("Transfered array of ") + std::to_string(intNumPoints) + std::string(" elements to GPU.")).c_str();
    while(m_time < TOTAL_TIME) {
        Log << (std::string("Start calculations at time: ") + std::to_string(m_time)).c_str();
        // TODO: updateBorders() gpu version
        // TODO: compute CUDA kernel
        // TODO: exchange halo points
        cu_loadBorderData(cu_gpuProp, m_borderElements, m_Ny, cu_loadFromDeviceToHost);
        // Debug -------------------
        #ifdef _DEBUG_
            writeFieldPart(m_borderElements, 2, m_Ny, Log, "Received border elements from GPU");
        #endif
        exchangeBorderPoints();
        cu_loadHaloData(cu_gpuProp, m_haloElements, m_Ny, cu_loadFromHostToDevice);
        // cudaDeviceSynchronize
        cu_loadFieldData(cu_gpuProp, m_Field, intNumPoints, cu_loadFromDeviceToHost);
        Log << (std::string("Transfered array of ") + std::to_string(intNumPoints) + std::string(" elements from GPU.")).c_str();
        sendBlockToServer();
        m_time += TAU;
    }
    setStopCheckMark();
    sendBlockToServer();
    Log << "Correct exit";
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

void ProcessNode::exchangeBorderPoints()
{
    /*MPI_Status status;
    int left_neighbor = (m_rank > 0) ? (m_rank - 1) : MPI_PROC_NULL;
    int right_neighbor = (m_rank < m_size - 2) ? (m_rank + 1) : MPI_PROC_NULL;
    int left_border_offset = m_bNy + 1; // skip the first boundary cell
    int right_border_offset = (m_columns - 1) * m_Ny + 1; // skip the first boundary cell
    int left_halo_offset = 0;
    int right_halo_offset = m_Ny;
    int num_halo_points = m_Ny;*/

    /* Send data to left, get data from right */
    //MPI_Sendrecv(m_haloElements + left_halo_offset, num_halo_points, MPI_CellType, left_neighbor, m_rank, h_right_halo,
    //    num_halo_points, MPI_CellType, right_neighbor, m_rank+1, MPI_COMM_WORLD, &status );
    /* Send data to right, get data from left */
    //MPI_Sendrecv(m_haloElements + right_halo_offset, num_halo_points, MPI_CellType, right_neighbor, m_rank, h_left_halo,
    //    num_halo_points, MPI_CellType, left_neighbor, m_rank-1, MPI_COMM_WORLD, &status );

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

void ProcessNode::setStopCheckMark()
{
    if(m_rank == 0) {
        m_Field[0].r = -1;
        Log << "'Stop' marker is set.";
    }
}
