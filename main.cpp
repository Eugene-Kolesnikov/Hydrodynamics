#include <mpi.h>
#include "processNode.h"
#include "serverNode.h"

int main(int argc, char** argv){

	int rank, size;
	int Nx = 100, Ny = 100;

	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank); // номер текущего процесса
    MPI_Comm_size (MPI_COMM_WORLD, &size); // число процессов

	if(rank < size - 1) { // computational nodes
		ProcessNode* process = new ProcessNode(rank,size,Nx,Ny);
		process->runNode();
		delete process;
	} else { // server node
		ServerNode* server = new ServerNode(rank,size,Nx,Ny);
		server->setArgcArgv(argc, argv);
		server->initDenseField();
		server->runNode();
		delete server;
	}

	MPI_Finalize();
	return 0;
}
