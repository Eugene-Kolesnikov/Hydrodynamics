Compilation:

```{bash}
cd ./guiPlot
qmake guiPlot.pro
make && make clean
cd ../
make && make clean
```

Execute the application:

```{bash}
mpirun -l -n 8 ./hydrodynamics
```