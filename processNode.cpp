#include "processNode.h"
#include <mpi.h>
#include "debug.h"

#define STEP_LENGTH 50

ProcessNode::ProcessNode(const int rank, const int size, const int Nx, const int Ny):
    Node::Node(rank, size, Nx, Ny),
    Log(rank, createLogFilename(rank))
{
    Log << std::string("GPU properties created.").c_str();
    int numCompNodes = m_size - 1;
    int numColumns = m_Nx;
    int intNumColumns = numColumns / numCompNodes;
    int intNumPoints = (intNumColumns + 2) * m_bNy; // (adding 2 columns of halo points)
    int numHaloElements = 2 * m_Ny;
    m_columns = (intNumColumns + 2);
    cu_gpuProp = cu_createGpuProperties(&Log, m_bNy, rank, size);
    m_Field = new Cell[intNumPoints];
    Log << (std::string("Allocated array of ") + std::to_string(intNumPoints) + std::string(" Field elements.")).c_str();
    m_borderElements = new Cell[numHaloElements]; // the number of border elements equal to the number of halo elements
    Log << (std::string("Allocated array of ") + std::to_string(numHaloElements) + std::string(" border elements.")).c_str();
    cu_allocateHostPinnedMemory((void**) &m_haloElements, numHaloElements, &Log);
    Log << (std::string("Allocated array of ") + std::to_string(numHaloElements) + std::string(" elements on Host in pinned memory using cudaHostAlloc.")).c_str();
    cu_allocateFieldMemory(cu_gpuProp, intNumPoints);
    Log << (std::string("Allocated array of ") + std::to_string(intNumPoints) + std::string(" Field elements on GPU.")).c_str();
}

ProcessNode::~ProcessNode()
{
    delete m_Field;
    cu_freeHostPinnedMemory(m_haloElements, &Log);
    cu_destroyGpuProperties(cu_gpuProp);
}

void ProcessNode::runNode()
{
    int intNumPoints = m_columns * m_bNy;
    initBlock();
    cu_loadFieldData(cu_gpuProp, m_Field, intNumPoints, cu_loadFromHostToDevice);
    int k = 0;
    while(m_time < TOTAL_TIME) {
        std::string time = std::to_string(m_time);
        Log << (std::string("Start calculations at time: ") + std::to_string(m_time)).c_str();
        cu_updateBorders(cu_gpuProp);
        #ifdef _DEBUG_
            cu_loadFieldData(cu_gpuProp, m_Field, intNumPoints, cu_loadFromDeviceToHost);
            writeFieldPart_id(m_Field, m_columns, m_bNy, 'r', Log, (time + "(after updating borders): Part of dense field with updated borders").c_str());
            writeFieldPart_id(m_Field, m_columns, m_bNy, 'u', Log, (time + "(after updating borders): Part of x-velocity field with updated borders").c_str());
            writeFieldPart_id(m_Field, m_columns, m_bNy, 'v', Log, (time + "(after updating borders): Part of y-velocity field with updated borders").c_str());
            writeFieldPart_id(m_Field, m_columns, m_bNy, 'e', Log, (time + "(after updating borders): Part of energy field with updated borders").c_str());
        #endif
        cu_computeBorderElements(cu_gpuProp);
        cu_loadBorderData(cu_gpuProp, m_borderElements, m_Ny, cu_loadFromDeviceToHost);
        #ifdef _DEBUG_
            writeFieldPart_id(m_borderElements, 2, m_Ny, 'r', Log, (time + ": Received border element's density from GPU").c_str());
            writeFieldPart_id(m_borderElements, 2, m_Ny, 'u', Log, (time + ": Received border element's from GPU").c_str());
            writeFieldPart_id(m_borderElements, 2, m_Ny, 'v', Log, (time + ": Received border element's from GPU").c_str());
            writeFieldPart_id(m_borderElements, 2, m_Ny, 'e', Log, (time + ": Received border element's from GPU").c_str());
        #endif
        cu_computeInternalElements(cu_gpuProp);
        #ifdef _DEBUG_
            cu_loadFieldData(cu_gpuProp, m_Field, intNumPoints, cu_loadFromDeviceToHost);
            writeFieldPart_id(m_Field, m_columns, m_bNy, 'r', Log, (time + "(after computing internal elements): Part of dense field with updated borders").c_str());
            writeFieldPart_id(m_Field, m_columns, m_bNy, 'u', Log, (time + "(after updating internal elements): Part of x-velocity field with updated borders").c_str());
            writeFieldPart_id(m_Field, m_columns, m_bNy, 'v', Log, (time + "(after updating internal elements): Part of y-velocity field with updated borders").c_str());
            writeFieldPart_id(m_Field, m_columns, m_bNy, 'e', Log, (time + "(after updating internal elements): Part of energy field with updated borders").c_str());
        #endif
        exchangeBorderPoints();
        cu_loadHaloData(cu_gpuProp, m_haloElements, m_Ny, cu_loadFromHostToDevice);
        /* `cu_loadFieldData` is performed with stream 'streamInternal' which doesn't require
           the stream 'streamHaloBorder' to be synchronized to start loading data. It requires only
           the computeBordersKernel to be finished which fulfills automatically because consequtive tasks
           of one stream permorm consequently */
        cu_moveBorderDataToField(cu_gpuProp);
        if(k % STEP_LENGTH == 0) {
            cu_loadFieldData(cu_gpuProp, m_Field, intNumPoints, cu_loadFromDeviceToHost);
            #ifdef _DEBUG_
                writeFieldPart_id(m_Field, m_columns, m_bNy, 'r', Log, (time + "(after moving border elements): Part of dense field with updated borders").c_str());
                writeFieldPart_id(m_Field, m_columns, m_bNy, 'u', Log, (time + "(after moving border elements): Part of x-velocity field with updated borders").c_str());
                writeFieldPart_id(m_Field, m_columns, m_bNy, 'v', Log, (time + "(after moving border elements): Part of y-velocity field with updated borders").c_str());
                writeFieldPart_id(m_Field, m_columns, m_bNy, 'e', Log, (time + "(after moving border elements): Part of energy field with updated borders").c_str());
            #endif
            cu_deviceSynchronize(); // whait while 'streamInternal' and 'streamHaloBorder' finish their work
            sendBlockToServer();
        } else
            cu_deviceSynchronize(); // whait while 'streamInternal' and 'streamHaloBorder' finish their work
        Log << "Device successfully synchronized.";
        m_time += TAU;
        ++k;
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
    MPI_Status status;
    int left_neighbor = (m_rank > 0) ? (m_rank - 1) : MPI_PROC_NULL;
    int right_neighbor = (m_rank < m_size - 2) ? (m_rank + 1) : MPI_PROC_NULL;
    int numPointsPerBorder = m_Ny;
    int offset[2] = {0, numPointsPerBorder};

    Log << (std::string("Try to send ") + std::to_string(numPointsPerBorder) +
            std::string(" amount of data to 'node ") + std::to_string(left_neighbor) +
            std::string("' and to recieve ") + std::to_string(numPointsPerBorder) +
            std::string(" amount of data from 'node ") + std::to_string(right_neighbor) + std::string("'")).c_str();
    /* Send left border to left node, get right halo from right node */
    MPI_Sendrecv(m_borderElements + offset[0], numPointsPerBorder, MPI_CellType, left_neighbor, m_rank,
                 m_haloElements + offset[1], numPointsPerBorder, MPI_CellType, right_neighbor, m_rank + 1, MPI_COMM_WORLD, &status );
    Log << "Data successfully sent and recieved.";

    Log << (std::string("Try to send ") + std::to_string(numPointsPerBorder) +
         std::string(" amount of data to 'node ") + std::to_string(left_neighbor) +
         std::string("' and to recieve ") + std::to_string(numPointsPerBorder) +
         std::string(" amount of data from 'node ") + std::to_string(right_neighbor) + std::string("'")).c_str();
    /* Send right border to right node, get left halo from left node */
    MPI_Sendrecv(m_borderElements + offset[1], numPointsPerBorder, MPI_CellType, right_neighbor, m_rank,
                 m_haloElements + offset[0], numPointsPerBorder, MPI_CellType, left_neighbor, m_rank - 1, MPI_COMM_WORLD, &status );
    Log << "Data successfully sent and recieved.";
    Log << "The process of exchanging halo points finished without errors.";
    #ifdef _DEBUG_
        writeFieldPart_id(m_haloElements, 2, m_Ny, 'r', Log, "Received halo elements from neighbor nodes");
    #endif
}

void ProcessNode::setStopCheckMark()
{
    if(m_rank == 0) {
        m_Field[0].r = -1;
        Log << "'Stop' marker is set.";
    }
}
