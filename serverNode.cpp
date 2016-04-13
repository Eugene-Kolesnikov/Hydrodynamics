#include "serverNode.h"
#include <mpi.h>
#include <iostream>

ServerNode::ServerNode(const int rank, const int size, const int Nx, const int Ny):
    Node::Node(rank, size, Nx, Ny)
{
    m_guiLibHandle = dlopen("guiPlot/libguiPlot.dylib", RTLD_LAZY);
    QplotField = (int (*)(int argc, char** argv, double** denseField, int Nx, int Ny))dlsym(m_guiLibHandle, "plotField");
}

ServerNode::~ServerNode()
{
    delete m_denseField;
}

void ServerNode::runNode()
{
    this->plotDenseField();
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
}

void ServerNode::shareInitField()
{

}

void ServerNode::loadUpdatedField()
{

}

void ServerNode::plotDenseField()
{
    QplotField(argc, argv, m_denseField, m_Nx, m_Ny);
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
    argc = t_argc;
    argv = t_argv;
}
