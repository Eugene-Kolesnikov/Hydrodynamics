#include "cell.h"
#include "LogSystem/FileLogger.hpp"
#include <string>

double debug_initNumber();
void writeFieldPart(Cell* M, int cols, int rows, logging::FileLogger& Log, std::string head);
void writeFieldPart_id(Cell* M, int cols, int rows, char ch, logging::FileLogger& Log, std::string head);
