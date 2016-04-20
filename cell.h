#ifndef CELL_H
#define CELL_H

#define cu_loadFromDeviceToHost 0
#define cu_loadFromHostToDevice 1

typedef struct {
    double r; // density
    double u; // x-velocity
    double v; // y-velocity
    double e; // energy
} Cell;

#endif
