#include <cuda.h>
#include "cu_gpuProperties.h"
#include "cell.h"
#include "cu_cell.hpp"

#include <stdio.h>

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
        }
        if(rank == totalRanks - 2) {
            halo2_tmp.r = border2->r; halo2_tmp.u = -border2->u;
            halo2_tmp.v = border2->v; halo2_tmp.e = border2->e;
            *halo2 = halo2_tmp;
        }
    }
}

__device__ cu_Cell cu_get_f(Cell* elem)
{
    Cell f;
    f.r = elem->r;
    f.u = elem->r * elem->u;
    f.v = elem->r * elem->v;
    f.e = elem->e + (pow(elem->u,2) + pow(elem->v,2))/2.0f;
    return cu_Cell(f);
}

__device__ cu_Cell cu_get_F(Cell* elem)
{
    Cell F;
    F.r = elem->r * elem->u;
    F.u = (GAMMA - 1) * elem->r * elem->e + elem->r * pow(elem->u,2);
    F.v = elem->r * elem->u * elem->v;
    F.e = GAMMA * elem->e + (pow(elem->u,2) + pow(elem->v,2))/2.0f;
    return cu_Cell(F);
}

__device__ cu_Cell cu_get_G(Cell* elem)
{
    Cell G;
    G.r = elem->r * elem->v;
    G.u = elem->r * elem->u * elem->v;
    G.v = (GAMMA - 1) * elem->r * elem->e + elem->r * pow(elem->v,2);
    G.e = GAMMA * elem->e + (pow(elem->u,2) + pow(elem->v,2))/2.0f;
    return cu_Cell(G);
}

__device__ Cell cu_get_elem(cu_Cell* f)
{
    Cell elem;
    elem.r = f->cell.r;
    elem.u = f->cell.u / elem.r;
    elem.v = f->cell.v / elem.r;
    elem.e = f->cell.e - (pow(elem.u,2)+pow(elem.v,2))/2.0f;
    return elem;
}

#ifdef _DEBUG_
__device__ Cell cu_get_elem_debug(cu_Cell* f)
{
    Cell elem;
    elem.r = f->cell.r;
    elem.u = f->cell.u;
    elem.v = f->cell.v;
    elem.e = f->cell.e;
    return elem;
}
#endif

__device__ void loadSharedMemory(Cell* field, int Ny, int fieldSize, int type)
{
    extern __shared__ Cell cellMemory[];

    if(type == _BORDERS_) {
        int tid = blockIdx.x * (blockDim.x - 2) + threadIdx.x;

        Cell* fieldPart = (threadIdx.y < 3) ? field : &field[fieldSize - 3 * (Ny + 2)];
        int column = (threadIdx.y < 3) ? threadIdx.y : threadIdx.y - 3;
        cellMemory[threadIdx.y * blockDim.x + threadIdx.x] = fieldPart[column * (Ny + 2) + tid];
        __syncthreads();
    } else {
        int tid_x = blockIdx.x * (blockDim.x - 2) + threadIdx.x;
        int tid_y = blockIdx.y * (blockDim.y - 2) + threadIdx.y;

        cellMemory[threadIdx.x * blockDim.y + threadIdx.y] = field[(tid_x + 1) * (Ny + 2) + tid_y];
        __syncthreads();
    }
}

__global__ void cu_computeElements(Cell* borders, Cell* field, int Nx, int Ny, int fieldSize, int type)
{
    extern __shared__ Cell cellMemory[];

    Cell* elem_ij; // elem{i,j}
    Cell* elem_ip1j; // elem{i+1,j}
    Cell* elem_im1j; // elem{i-1,j}
    Cell* elem_ijp1; // elem{i,j+1}
    Cell* elem_ijm1; // elem{i,j-1}
    Cell* cell;

    if(type == _BORDERS_) {
        int tid = blockIdx.x * (blockDim.x - 2) + threadIdx.x;

        if(tid >= Ny+2) // eliminate unnecessary threads
            return;

        loadSharedMemory(field, Ny, fieldSize, type);

        if(tid >= Ny || threadIdx.x >= 30 || threadIdx.y > 1) // eliminate unnecessary threads
            return;

        // get the adresses of necessary cells
        Cell* cellMemoryPart = (threadIdx.y == 0) ? &cellMemory[0*blockDim.x] : &cellMemory[3*blockDim.x];
        cell = (threadIdx.y == 0) ? &borders[tid] : &borders[Ny + tid];
        elem_ij   = &cellMemoryPart[1*blockDim.x + 1 + threadIdx.x]; // elem{i,j}
        elem_ip1j = &cellMemoryPart[2*blockDim.x + 1 + threadIdx.x]; // elem{i+1,j}
        elem_im1j = &cellMemoryPart[0*blockDim.x + 1 + threadIdx.x]; // elem{i-1,j}
        elem_ijp1 = &cellMemoryPart[1*blockDim.x + 1 + threadIdx.x+1]; // elem{i,j+1}
        elem_ijm1 = &cellMemoryPart[1*blockDim.x + 1 + threadIdx.x-1]; // elem{i,j-1}
    } else if(type == _INTERNAL_) {
        int tid_x = blockIdx.x * (blockDim.x - 2) + threadIdx.x;
        int tid_y = blockIdx.y * (blockDim.y - 2) + threadIdx.y;

        if(tid_x >= Nx || tid_y >= Ny+2)
            return;

        loadSharedMemory(field, Ny, fieldSize, type);

        if(tid_x >= Nx-2 || tid_y >= Ny || threadIdx.x >= 30 || threadIdx.y >= 30)
            return;

        // get the adresses of necessary cells
        elem_ij   = &cellMemory[(threadIdx.x + 1) * blockDim.y + 1 + threadIdx.y]; // elem{i,j}
        elem_ip1j = &cellMemory[(threadIdx.x + 2) * blockDim.y + 1 + threadIdx.y]; // elem{i+1,j}
        elem_im1j = &cellMemory[(threadIdx.x + 0) * blockDim.y + 1 + threadIdx.y]; // elem{i-1,j}
        elem_ijp1 = &cellMemory[(threadIdx.x + 1) * blockDim.y + 1 + threadIdx.y + 1]; // elem{i,j+1}
        elem_ijm1 = &cellMemory[(threadIdx.x + 1) * blockDim.y + 1 + threadIdx.y - 1]; // elem{i,j-1}
        cell = &field[(tid_x+2)*(Ny + 2) + 1 + tid_y];
    }

    cu_Cell f_ij   = cu_get_f(elem_ij); // f{i,j}
    cu_Cell f_ip1j = cu_get_f(elem_ip1j); // f{i+1,j}
    cu_Cell f_ijp1 = cu_get_f(elem_ijp1); // f{i,j+1}
    cu_Cell f_im1j = cu_get_f(elem_im1j); // f{i-1,j}
    cu_Cell f_ijm1 = cu_get_f(elem_ijm1); // f{i,j-1}

    cu_Cell F_ij   = cu_get_F(elem_ij); // F{i,j}
    cu_Cell F_ip1j = cu_get_F(elem_ip1j); // F{i+1,j}
    cu_Cell F_im1j = cu_get_F(elem_im1j); // F{i-1,j}

    cu_Cell G_ij   = cu_get_G(elem_ij); // G{i,j}
    cu_Cell G_ijp1 = cu_get_G(elem_ijp1); // G{i,j+1}
    cu_Cell G_ijm1 = cu_get_G(elem_ijm1); // G{i,j-1}

    float D_ip1j = max(fabs(elem_ij->u) + sqrt(10/9 * elem_ij->e),
                       fabs(elem_ip1j->u) + sqrt(10/9 * elem_ip1j->e)); // D{i+1,j}
    float D_im1j = max(fabs(elem_ij->u) + sqrt(10/9 * elem_ij->e),
                       fabs(elem_im1j->u) + sqrt(10/9 * elem_im1j->e)); // D{i-1,j}
    float D_ijp1 = max(fabs(elem_ij->v) + sqrt(10/9 * elem_ij->e),
                       fabs(elem_ijp1->v) + sqrt(10/9 * elem_ijp1->e)); // D{i,j+1}
    float D_ijm1 = max(fabs(elem_ij->v) + sqrt(10/9 * elem_ij->e),
                       fabs(elem_ijm1->v) + sqrt(10/9 * elem_ijm1->e)); // D{i,j-1}

    cu_Cell F_ip12j = (F_ip1j + F_ij - (f_ip1j - f_ij) * D_ip1j) * 0.5f; // F{i+1/2,j}
    cu_Cell F_im12j = (F_ij + F_im1j - (f_ij - f_im1j) * D_im1j) * 0.5f; // F{i-1/2,j}
    cu_Cell G_ijp12 = (G_ijp1 + G_ij - (f_ijp1 - f_ij) * D_ijp1) * 0.5f; // G{i,j+1/2}
    cu_Cell G_ijm12 = (G_ij + G_ijm1 - (f_ij - f_ijm1) * D_ijm1) * 0.5f; // G{i,j-1/2}

    cu_Cell f_new = f_ij - (( F_ip12j - F_im12j ) * Nx + ( G_ijp12 - G_ijm12 ) * Ny) * TAU;

    __syncthreads();

    *cell = cu_get_elem(&f_new);
    //*cell = cu_get_elem_debug(&f_new);
    //*cell = *elem_ij;
}
