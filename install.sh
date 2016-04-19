#!/usr/bin/env bash
rm -r build
cd ./guiPlot &&
qmake guiPlot.pro &&
make && make clean &&
rm libguiPlot.dylib libguiPlot.1.dylib libguiPlot.1.0.dylib &&
cd ../ &&
make && make clean &&
mkdir build &&
cp ./guiPlot/libguiPlot.1.0.0.dylib ./build/ &&
cp ./hydrodynamics ./build/ &&
cd ./build/ &&
touch execute.sh &&
chmod 777 execute.sh &&
echo "#!/usr/bin/env bash" >> ./execute.sh &&
echo "mpiexec -l -np 5 ./hydrodynamics -gui libguiPlot.1.0.0.dylib -nx 100 -ny 100" >> ./execute.sh &&
mkdir img log
