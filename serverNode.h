#ifndef SERVERNODE_H
#define SERVERNODE_H

#include <dlfcn.h>
#include "guiPlot/guiplot.h"
#include "node.h"
#include <string>
#include "LogSystem/FileLogger.hpp"

class ServerNode : public Node
{
public:
    ServerNode(const int rank, const int size, const int Nx, const int Ny);
    ~ServerNode();
    void runNode();
    void setArgcArgv(int t_argc, char** t_argv);
    void loadGui(std::string gui_dl);

private:
    void initDenseField();
    void shareInitField();
    void loadUpdatedField();
    void plotDenseField();

    void cellToCoord(int xIndex, int yIndex, double* x, double* y);
    std::string getFilename();
    void parseCmdArgv_guiDL(int argc, char** argv);

private:
    int m_argc;
    char** m_argv;
    bool m_loadedGui;
    void* m_guiLibHandle;
    int (*QplotField)(int argc, char** argv, double** denseField, int Nx, int Ny, const char* filename);

private:
    double** m_denseField;
    int m_fileCount;
    logging::FileLogger Log;
};

#endif // SERVERNODE_H
