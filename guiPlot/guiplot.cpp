#include "guiplot.h"
#include <QApplication>
#include "qcustomplot.h"
#include "../cell.h"

extern "C"
int plotField(int argc, char** argv, void* arr, int Nx, int Ny, const char* filename, char type)
{
    QApplication a(argc, argv);
    QCustomPlot* customPlot = new QCustomPlot();
    // configure axis rect:
    customPlot->axisRect()->setupFullAxesBox(true);
    customPlot->xAxis->setLabel("x");
    customPlot->yAxis->setLabel("y");

    int bNx = Nx + 2;
    int bNy = Ny + 2;

    // set up the QCPColorMap:
    QCPColorMap *colorMap = new QCPColorMap(customPlot->xAxis, customPlot->yAxis);
    customPlot->addPlottable(colorMap);

    colorMap->data()->setSize(Nx, Ny); // we want the color map to have nx * ny data points
    colorMap->data()->setRange(QCPRange(0, 1), QCPRange(0, 1)); // and span the coordinate range 0..1 in both key (x) and value (y) dimensions
    // now we assign some data, by accessing the QCPColorMapData instance of the color map:
    Cell* array = (Cell*)arr;
    for (int xIndex = 1; xIndex < bNx - 1; ++xIndex) {
        for (int yIndex = 1; yIndex < bNy - 1; ++yIndex) {
            switch (type) {
                case 'r': colorMap->data()->setCell(xIndex-1, yIndex-1, array[xIndex * bNy + yIndex].r); break;
                case 'u': colorMap->data()->setCell(xIndex-1, yIndex-1, array[xIndex * bNy + yIndex].u); break;
                case 'v': colorMap->data()->setCell(xIndex-1, yIndex-1, array[xIndex * bNy + yIndex].v); break;
                case 'e': colorMap->data()->setCell(xIndex-1, yIndex-1, array[xIndex * bNy + yIndex].e); break;
            }
        }
    }

    switch (type) {
        case 'r':
            colorMap->data()->setCell(0, 0, 42); //hack!
            break;
        case 'u':
            colorMap->data()->setCell(0, 0, -0.32); //hack!
            colorMap->data()->setCell(Nx-1, Ny-1, 0.32); //hack!
            break;
        case 'v':
            colorMap->data()->setCell(0, 0, -0.8); //hack!
            colorMap->data()->setCell(Nx-1, Ny-1, 0.8); //hack!
            break;
        case 'e':
            colorMap->data()->setCell(0, 0, 0); //hack!
            colorMap->data()->setCell(Nx-1, Ny-1, 2.5); //hack!
            break;
    }

    // add a color scale:
    QCPColorScale *colorScale = new QCPColorScale(customPlot);
    customPlot->plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect
    colorScale->setType(QCPAxis::atRight); // scale shall be vertical bar with tick/axis labels right (actually atRight is already the default)
    colorMap->setColorScale(colorScale); // associate the color map with the color scale
    switch (type) {
        case 'r': colorScale->axis()->setLabel("Density Field"); break;
        case 'u': colorScale->axis()->setLabel("X-velocity Field"); break;
        case 'v': colorScale->axis()->setLabel("Y-velocity Field"); break;
        case 'e': colorScale->axis()->setLabel("Energy Field"); break;
    }

    // set the color gradient of the color map to one of the presets:
    colorMap->setGradient(QCPColorGradient::gpJet); // gpJet, gpThermal
    // we could have also created a QCPColorGradient instance and added own colors to
    // the gradient, see the documentation of QCPColorGradient for what's possible.

    // rescale the data dimension (color) such that all data points lie in the span visualized by the color gradient:
    colorMap->rescaleDataRange();

    // make sure the axis rect and color scale synchronize their bottom and top margins (so they line up):
    QCPMarginGroup *marginGroup = new QCPMarginGroup(customPlot);
    customPlot->axisRect()->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);
    colorScale->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);

    // rescale the key (x) and value (y) axes so the whole color map is visible:
    customPlot->rescaleAxes();
    customPlot->savePng(filename,1280,1024);

    delete customPlot;
    return 0;
}
