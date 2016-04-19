#include "serverNode.h"
#include <mpi.h>
#include <iostream>
#include <getopt.h>

ServerNode::ServerNode(const int rank, const int size, const int Nx, const int Ny):
    Node::Node(rank, size, Nx, Ny), m_loadedGui(false), m_fileCount(0),
    Log(rank, createLogFilename(rank))
{
    initDenseField();
}

ServerNode::~ServerNode()
{
    if(m_loadedGui == true)
        dlclose(m_guiLibHandle);
    delete m_Field;
}

void ServerNode::runNode()
{
    //for(int i = 0; i < 11; ++i) {
    shareInitField();
    loadUpdatedField();
    plotDenseField();
    Log << "Plotted dense field";
    //}
}

void ServerNode::initDenseField()
{
    int id;
    double x, y;
    double d1 = 1.0 / 3.0;
    double d2 = 2.0 / 3.0;
    double del = 0.1;
    m_Field = new Cell [m_Nx * m_Ny];
    for (int xIndex = 0; xIndex < m_Nx; ++xIndex)
    {
        for (int yIndex = 0; yIndex < m_Ny; ++yIndex)
        {
            covert2Dto1D(xIndex, yIndex, &id);
            cellToCoord(xIndex, yIndex, &x, &y);
            if(y < d1 - del) {
                m_Field[id].r = 1;
                m_Field[id].u = 1;
                m_Field[id].v = 1;
                m_Field[id].e = 1;
            } else if(y < d1) {
                if(x < d1 || x > 2.0*d1) {
                    m_Field[id].r = 1;
                    m_Field[id].u = 1;
                    m_Field[id].v = 1;
                    m_Field[id].e = 1;
                } else {
                    m_Field[id].r = 5;
                    m_Field[id].u = 5;
                    m_Field[id].v = 5;
                    m_Field[id].e = 5;
                }
            } else if(y < d2) {
                m_Field[id].r = 5;
                m_Field[id].u = 5;
                m_Field[id].v = 5;
                m_Field[id].e = 5;
            } else {
                m_Field[id].r = 7;
                m_Field[id].u = 7;
                m_Field[id].v = 7;
                m_Field[id].e = 7;
            }
        }
    }
    Log << "Dense field initialized.";
}

void ServerNode::shareInitField()
{
    Log << "Sharing the initialized dense field.";
    // devide the space into `numCompNodes` consequtive blocks along X-axis
    int numCompNodes = m_size - 1; // last one is a server node
    int first_node = 0;
    int last_node = m_size - 2;
    int numColumns = m_Nx;
    int intNumColumns = numColumns / numCompNodes;
    int intNumPoints = (intNumColumns + 2) * m_Ny; // (adding 2 columns of halo points)
    int edgeNumPoints = (intNumColumns + 1) * m_Ny; // (adding 1 column of halo points)
    int intNumShift = intNumColumns * m_Ny; // `send_address` shift for internal blocks
    int edgeNumShift = (intNumColumns - 1) * m_Ny; // `send_address` shift for the first block
    Cell* send_address = m_Field;

    // sending data to other nodes
    MPI_Send(send_address, edgeNumPoints, MPI_CellType, first_node, 0, MPI_COMM_WORLD );
    send_address += edgeNumShift;
    Log << "Data sent to the 0 node.";
    for(int node = first_node + 1; node < last_node; ++node) {
        MPI_Send(send_address, intNumPoints, MPI_CellType, node, 0, MPI_COMM_WORLD );
        send_address += intNumShift;
        Log << (std::string("Data sent to the ") + std::to_string(node) + std::string(" node.")).c_str();
    }
    MPI_Send(send_address, edgeNumPoints, MPI_CellType, last_node, 0, MPI_COMM_WORLD );
    Log << (std::string("Data sent to the ") + std::to_string(last_node) + std::string(" node.")).c_str();

    // make sure that every one has it's par of data
    MPI_Barrier(MPI_COMM_WORLD);
    Log << "Data successfully sent to all nodes and barrier syncronization performed.";
}

void ServerNode::loadUpdatedField()
{
    int numCompNodes = m_size - 1;
    int first_node = 0;
    int last_node = m_size - 2;
    int numColumns = m_Nx;
    int intNumColumns = numColumns / numCompNodes;
    int intNumShift = intNumColumns * m_Ny;

    MPI_Status status;
    Cell* recv_address = m_Field;
    for(int node = first_node; node <= last_node; ++node) {
        MPI_Recv(recv_address, intNumShift, MPI_CellType, node, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
        recv_address += intNumShift;
        Log << (std::string("Data recieved from the ") + std::to_string(node) + std::string(" node.")).c_str();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    Log << "Data successfully recieved from all nodes and barrier synchronization performed.";
}

void ServerNode::plotDenseField()
{
    if(m_loadedGui == true) {
        std::string filename = getFilename();
        QplotField(m_argc, m_argv, (void*)m_Field, m_Nx, m_Ny, filename.c_str());
        ++m_fileCount;
    }
}

void ServerNode::cellToCoord(int xIndex, int yIndex, double* x, double* y)
{
    double hx = 1.0 / static_cast<double>(m_Nx);
    double hy = 1.0 / static_cast<double>(m_Ny);
    *x = hx * xIndex;
    *y = hy * yIndex;
}

void ServerNode::covert2Dto1D(int xIndex, int yIndex, int* id)
{
    *id = xIndex * m_Ny + yIndex;
}

void ServerNode::setArgcArgv(int t_argc, char** t_argv)
{
    m_argc = t_argc;
    m_argv = t_argv;
}

void ServerNode::loadGui(std::string gui_dl)
{
    bool noErrors = true;
    m_guiLibHandle = dlopen(gui_dl.c_str(), RTLD_LAZY);
    if (!m_guiLibHandle) {
        fputs (dlerror(), stderr);
        noErrors = false;
    }
    QplotField = (int (*)(int argc, char** argv, void* Field, int Nx, int Ny, const char* filename))dlsym(m_guiLibHandle, "plotField");
    char *error;
    if (noErrors == true && (error = dlerror()) != NULL) {
        fputs(error, stderr);
        noErrors = false;
        delete error;
    }
    if(noErrors == true) {
        m_loadedGui = true;
    }
}

std::string ServerNode::getFilename()
{
    std::string filename("img/denseField");
    if(m_fileCount < 10) {
        filename += "000";
    } else if(m_fileCount < 100) {
        filename += "00";
    } else if(m_fileCount < 1000) {
        filename += "0";
    }
    filename += (std::to_string(m_fileCount) + ".png");
    //std::cout << filename << std::endl;
    return filename;
}
