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

void writeFieldPart_id(Cell* M, int cols, int rows, char ch, logging::FileLogger& Log, std::string head)
{
    std::string str = head + ":\n";
    for(int j = 0; j < rows; ++j) {
        for(int i = 0; i < cols; ++i) {
            if(ch == 'r') {
                str += (std::to_string(M[i*rows + j].r) + " ");
            } else if(ch == 'u') {
                str += (std::to_string(M[i*rows + j].u) + " ");
            } else if(ch == 'v') {
                str += (std::to_string(M[i*rows + j].v) + " ");
            } else if(ch == 'e') {
                str += (std::to_string(M[i*rows + j].e) + " ");
            }
        }
        str += "\n";
    }
    str += "\n";
    Log << str.c_str();
}
