#ifndef SERVERNODE_H
#define SERVERNODE_H

#include <dlfcn.h>
#include "guiPlot/guiplot.h"
#include "node.h"

class ServerNode : public Node
{
public:
    ServerNode(const int rank, const int size, const int Nx, const int Ny);
    ~ServerNode();

    void runNode();

    void setArgcArgv(int t_argc, char** t_argv);
    void initDenseField();

private:
    void shareInitField();
    void loadUpdatedField();
    void plotDenseField();

    void cellToCoord(int xIndex, int yIndex, double* x, double* y);

private:
    double** m_denseField;
    void* m_guiLibHandle;
    int (*QplotField)(int argc, char** argv, double** denseField, int Nx, int Ny);
    int argc;
    char** argv;
};

#endif // SERVERNODE_H
