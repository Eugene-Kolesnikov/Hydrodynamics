#include "guiplot.h"
#include <QApplication>
#include "qcustomplot.h"

extern "C"
int plotField(int argc, char** argv, double* array, int Nx, int Ny, const char* filename)
{
    QApplication a(argc, argv);
    QCustomPlot* customPlot = new QCustomPlot();
    // configure axis rect:
    customPlot->axisRect()->setupFullAxesBox(true);
    customPlot->xAxis->setLabel("x");
    customPlot->yAxis->setLabel("y");

    // set up the QCPColorMap:
    QCPColorMap *colorMap = new QCPColorMap(customPlot->xAxis, customPlot->yAxis);
    customPlot->addPlottable(colorMap);
    int nx = static_cast<int>(Nx);
    int ny = static_cast<int>(Ny);
    colorMap->data()->setSize(nx, ny); // we want the color map to have nx * ny data points
    colorMap->data()->setRange(QCPRange(0, 1), QCPRange(0, 1)); // and span the coordinate range 0..1 in both key (x) and value (y) dimensions
    // now we assign some data, by accessing the QCPColorMapData instance of the color map:
    for (int xIndex=0; xIndex<nx; ++xIndex)
        for (int yIndex=0; yIndex<ny; ++yIndex)
            colorMap->data()->setCell(xIndex, yIndex, array[xIndex*nx + yIndex]);

    colorMap->data()->setCell(0, 0, 10); //hack!

    // add a color scale:
    QCPColorScale *colorScale = new QCPColorScale(customPlot);
    customPlot->plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect
    colorScale->setType(QCPAxis::atRight); // scale shall be vertical bar with tick/axis labels right (actually atRight is already the default)
    colorMap->setColorScale(colorScale); // associate the color map with the color scale
    colorScale->axis()->setLabel("Dense Field");

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
    customPlot->savePng(filename,640,480);

    delete customPlot;
    return 0;
}
