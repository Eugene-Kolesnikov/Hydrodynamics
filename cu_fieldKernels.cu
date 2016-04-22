#include <cuda.h>
#include "cu_gpuProperties.h"
#include "cell.h"

__global__ void updateBordersKernel(Cell* field, int Nx, int Ny, char type, int rank, int totalRanks)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Cell* border1, *border2;
    Cell* halo1, *halo2;
    int fieldSize = (Nx + 2) * (Ny + 2);
    Cell halo1_tmp;
    Cell halo2_tmp;
    if(type == 'h') { // horizontal
        if(tid >= Nx)
            return;
        border1 = field + (1 + tid) * (Ny + 2) + 1;
        halo1 = field + (1 + tid) * (Ny + 2);
        border2 = field + (1 + tid) * (Ny + 2) + Ny;
        halo2 = field + (1 + tid) * (Ny + 2) + Ny + 1;

        halo1_tmp.r = border1->r; halo1_tmp.u = border1->u;
        halo1_tmp.v = -border1->v; halo1_tmp.e = border1->e;
        halo2_tmp.r = border2->r; halo2_tmp.u = border2->u;
        halo2_tmp.v = -border2->v; halo2_tmp.e = border2->e;

        *halo1 = halo1_tmp;
        *halo2 = halo2_tmp;
    } else { // vertical
        if(tid >= Ny || (rank != 0 && rank != totalRanks - 2))
            return;
        border1 = field + (Ny + 2) + 1 + tid;
        halo1 = field + 1 + tid;
        border2 = field + fieldSize - 2*(Ny + 2) + 1 + tid;
        halo2 = field + fieldSize - (Ny + 2) + 1 + tid;

        if(rank == 0) {
            halo1_tmp.r = border1->r; halo1_tmp.u = -border1->u;
            halo1_tmp.v = border1->v; halo1_tmp.e = border1->e;
            *halo1 = halo1_tmp;
        } else if(rank == totalRanks - 2) {
            halo2_tmp.r = border2->r; halo2_tmp.u = -border2->u;
            halo2_tmp.v = border2->v; halo2_tmp.e = border2->e;
            *halo2 = halo2_tmp;
        }
    }
}
