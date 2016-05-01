#ifndef CELL_H
#define CELL_H

//#define _DEBUG_

#define cu_loadFromDeviceToHost 0
#define cu_loadFromHostToDevice 1

#define GAMMA (5.0f/3.0f)
// how long program will work
#define TOTAL_TIME 1.0e-1 //+2
// time step
#define TAU 1.0e-4

struct Cell {
    double r; // density
    double u; // x-velocity
    double v; // y-velocity
    double e; // energy
};

#endif
