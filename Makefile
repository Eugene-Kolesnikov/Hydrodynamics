NVCC        = nvcc
MPI_CXX     = mpicxx

LIBS_PATH = -L/usr/local/Cellar/mpich/3.2/lib/
LIBS =  -lmpi -lopa -lmpl -lrt -lcr -lpthread
INCLUDE_PATH = -I/usr/local/Cellar/mpich/3.2/include/
CXXFLAGS = -c -O3 -std=c++11
CFLAGS = -c
NVCCFLAGS = -arch=sm_30 -O3 -ccbin=$(MPI_CXX) -std=c++11

OBJECTS = main.o processNode.o serverNode.o LogSystem/FileLogger.o transferAllocation.o cu_gpuProperties.o

all: $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(OBJECTS) -o hydrodynamics

%.o: %.cpp
	$(MPI_CXX) $(CXXFLAGS) $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) $< -o $@

clean:
	rm *.o
	rm LogSystem/*.o
