#-------------------------------------------------
#
# Project created by QtCreator 2016-04-13T23:52:53
#
#-------------------------------------------------

QT       += widgets

greaterThan(QT_MAJOR_VERSION, 4): QT += printsupport

TARGET = guiPlot
TEMPLATE = lib

DEFINES += GUIPLOT_LIBRARY

SOURCES += guiplot.cpp \
    qcustomplot.cpp

HEADERS += guiplot.h \
    qcustomplot.h

unix {
    target.path = /usr/lib
    INSTALLS += target
}
