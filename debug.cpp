#include "debug.h"

double debug_initNumber()
{
    static int val = 10;
    return val++;
}

void writeFieldPart(Cell* M, int cols, int rows, logging::FileLogger& Log, std::string head)
{
    std::string str = head + ":\n";
    for(int j = 0; j < rows; ++j) {
        for(int i = 0; i < cols; ++i) {
            str += (std::to_string(M[i*rows + j].u) + " ");
        }
        str += "\n";
    }
    Log << str.c_str();
}
