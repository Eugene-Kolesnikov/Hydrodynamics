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
echo "mpiexec -l -np 5 ./hydrodynamics -gui libguiPlot.1.0.0.dylib -nx 640 -ny 640 && ./createVideos.sh" >> ./execute.sh &&
touch createVideos.sh &&
chmod 777 createVideos.sh &&
echo "#!/usr/bin/env bash" >> ./createVideos.sh &&
echo "ffmpeg -framerate 60 -i img/densityField%04d.png -c:v libx264 -r 60 -pix_fmt yuv420p densityField.mp4" >> ./createVideos.sh &&
echo "ffmpeg -framerate 60 -i img/xVelField%04d.png -c:v libx264 -r 60 -pix_fmt yuv420p xVelField.mp4" >> ./createVideos.sh &&
echo "ffmpeg -framerate 60 -i img/yVelField%04d.png -c:v libx264 -r 60 -pix_fmt yuv420p yVelField.mp4" >> ./createVideos.sh &&
echo "ffmpeg -framerate 60 -i img/energyField%04d.png -c:v libx264 -r 60 -pix_fmt yuv420p energyField.mp4" >> ./createVideos.sh &&
mkdir img log
