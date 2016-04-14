#include <mpi.h>
#include "processNode.h"
#include "serverNode.h"
#include <getopt.h>
#include <stdlib.h>

void parseCmdArgv(int argc, char** argv, int* Nx, int* Ny, std::string& gui_dl);

int rank, size;

int main(int argc, char** argv){

	int Nx = -1, Ny = -1;
	std::string gui_dl;

	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank); // номер текущего процесса
    MPI_Comm_size (MPI_COMM_WORLD, &size); // число процессов

	parseCmdArgv(argc, argv, &Nx, &Ny, gui_dl);
	if((Nx == -1 || Ny == -1) && rank == size - 1) {
		fputs("Error: not enough cmd arguments!\n", stderr);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if(rank < size - 1) { // computational nodes
		ProcessNode* process = new ProcessNode(rank,size,Nx,Ny);
		process->runNode();
		delete process;
	} else { // server node
		ServerNode* server = new ServerNode(rank,size,Nx,Ny);
		server->setArgcArgv(argc, argv);
		if(gui_dl.empty() == false)
			server->loadGui(gui_dl);
		server->runNode();
		delete server;
	}

	MPI_Finalize();
	return 0;
}

void parseCmdArgv(int argc, char** argv, int* Nx, int* Ny, std::string& gui_dl)
{
	int c;
    int digit_optind = 0;
	std::string nx("nx");
	std::string ny("ny");
	std::string gui("gui");

	while (1) {
    	int this_option_optind = optind ? optind : 1;
        int option_index = 0;
        static struct option long_options[] = {
            {"gui", required_argument, 0,  0 },
            {"nx",  required_argument, 0,  0 },
            {"ny",  required_argument, 0,  0 },
            {0,         0,             0,  0 }
        };

       c = getopt_long_only(argc, argv, "abc:d:012", long_options, &option_index);

	   if (c == -1)
	   		break;

	   	std::string opt = long_options[option_index].name;
       	switch (c) {
		    case 0:
				if(opt == nx)
					*Nx = atoi(optarg);
				else if(opt == ny)
					*Ny = atoi(optarg);
				else if(opt == gui)
					gui_dl = std::string(optarg);
				//printf("option %s", long_options[option_index].name);
				//if (optarg)
				//	printf(" with arg %s", optarg);
				//printf("\n");
			default:
            	break;
        }
    }
}
