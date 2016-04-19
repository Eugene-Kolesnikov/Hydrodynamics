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
    shareInitField();
    loadUpdatedField();
    plotDenseField();

    Log << "Plotted dense field";
}

void ServerNode::initDenseField()
{
    int id;
    double x, y;
    double d1 = 1.0 / 3.0;
    double d2 = 2.0 / 3.0;
    double del = 0.1;
    m_Field = new Cell [m_bNx * m_bNy];
    for (int xIndex = 1; xIndex < m_bNx-1; ++xIndex)
    {
        for (int yIndex = 1; yIndex < m_bNy-1; ++yIndex)
        {
            id = covert2Dto1D(xIndex, yIndex);
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
    int numActualColumns = m_Nx;
    int numColumns = numActualColumns / numCompNodes;
    int numPoints = (numColumns + 2) * m_bNy; // (adding 2 columns of halo points)
    int numShift = numColumns * m_bNy; // `send_address` shift for internal blocks
    Cell* send_address = m_Field;

    // sending data to other nodes
    for(int node = first_node; node <= last_node; ++node) {
        Log << (std::string("Try to send ") + std::to_string(numPoints) + std::string(" amount of data to node ") + std::to_string(node)).c_str();
        MPI_Send(send_address, numPoints, MPI_CellType, node, 0, MPI_COMM_WORLD );
        send_address += numShift;
        Log << (std::string("Data sent to the node ") + std::to_string(node)).c_str();
    }

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
    int intNumShift = intNumColumns * m_bNy;

    MPI_Status status;
    Cell* recv_address = m_Field;
    for(int node = first_node; node <= last_node; ++node) {
        int num = intNumShift + (node == first_node || node == last_node) * m_bNy;
        Log << (std::string("Try to recieve ") + std::to_string(num) + std::string(" amount of data from node ") + std::to_string(node)).c_str();
        MPI_Recv(recv_address, num, MPI_CellType, node, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
        recv_address += num;
        Log << (std::string("Data recieved from the node ") + std::to_string(node)).c_str();
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
    *x = hx * (xIndex-1);
    *y = hy * (yIndex-1);
}

int ServerNode::covert2Dto1D(int xIndex, int yIndex)
{
    return xIndex * m_bNy + yIndex;
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
    return filename;
}
