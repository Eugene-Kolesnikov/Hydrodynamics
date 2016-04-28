#ifndef CU_CELL_H
#define CU_CELL_H

#include "cell.h"

struct cu_Cell {
    Cell cell;

    __device__ cu_Cell(){}
    __device__ cu_Cell(Cell c): cell(c) {}

    friend __device__ cu_Cell operator+(cu_Cell lhs, const cu_Cell& rhs) {
        lhs.cell.r += rhs.cell.r;
        lhs.cell.u += rhs.cell.u;
        lhs.cell.v += rhs.cell.v;
        lhs.cell.e += rhs.cell.e;
        return lhs;
    }

    friend __device__ cu_Cell operator-(cu_Cell lhs, const cu_Cell& rhs) {
        lhs.cell.r -= rhs.cell.r;
        lhs.cell.u -= rhs.cell.u;
        lhs.cell.v -= rhs.cell.v;
        lhs.cell.e -= rhs.cell.e;
        return lhs;
    }

    friend __device__ cu_Cell operator*(cu_Cell lhs, float val) {
        lhs.cell.r *= val;
        lhs.cell.u *= val;
        lhs.cell.v *= val;
        lhs.cell.e *= val;
        return lhs;
    }

    friend __device__ cu_Cell operator/(cu_Cell lhs, float val) {
        lhs.cell.r /= val;
        lhs.cell.u /= val;
        lhs.cell.v /= val;
        lhs.cell.e /= val;
        return lhs;
    }
};

#endif
