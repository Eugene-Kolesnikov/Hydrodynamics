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
    delete m_denseField;
}

void ServerNode::runNode()
{
    for(int i = 0; i < 11; ++i) {
        this->plotDenseField();
        Log << ("Plotted field N" + std::to_string(i)).c_str();
    }
}

void ServerNode::initDenseField()
{
    double x, y;
    double d1 = 1.0 / 3.0;
    double d2 = 2.0 / 3.0;
    double del = 0.1;
    m_denseField = new double* [m_Nx];
    for (int xIndex = 0; xIndex < m_Nx; ++xIndex)
    {
        m_denseField[xIndex] = new double [m_Ny];
        for (int yIndex = 0; yIndex < m_Ny; ++yIndex)
        {
            cellToCoord(xIndex, yIndex, &x, &y);
            if(y < d1 - del) {
                m_denseField[xIndex][yIndex] = 1;
            } else if(y < d1) {
                if(x < d1 || x > 2.0*d1)
                    m_denseField[xIndex][yIndex] = 1;
                else
                    m_denseField[xIndex][yIndex] = 5;
            } else if(y < d2) {
                m_denseField[xIndex][yIndex] = 5;
            } else {
                m_denseField[xIndex][yIndex] = 7;
            }
        }
    }
    Log << "Dense field initialized.";
}

void ServerNode::shareInitField()
{

}

void ServerNode::loadUpdatedField()
{

}

void ServerNode::plotDenseField()
{
    if(m_loadedGui == true) {
        std::string filename = getFilename();
        QplotField(m_argc, m_argv, m_denseField, m_Nx, m_Ny, filename.c_str());
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
    QplotField = (int (*)(int argc, char** argv, double** denseField, int Nx, int Ny, const char* filename))dlsym(m_guiLibHandle, "plotField");
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
    std::string filename("denseField");
    if(m_fileCount < 10) {
        filename += "000";
    } else if(m_fileCount < 100) {
        filename += "00";
    } else if(m_fileCount < 1000) {
        filename += "0";
    }
    filename += (std::to_string(m_fileCount) + ".png");
    std::cout << filename << std::endl;
    return filename;
}
