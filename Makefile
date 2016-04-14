NVCC       = nvcc
MPI_CC     = mpicxx

LIBS_PATH = -L/usr/local/Cellar/mpich/3.2/lib/
LIBS =  -lmpi -lopa -lmpl -lrt -lcr -lpthread
INCLUDE_PATH = -I/usr/local/Cellar/mpich/3.2/include/
CFLAGS = -c -O3
NVCCFLAGS = -arch=sm_30 -O3 -ccbin=$(MPI_CC)

OBJECTS = main.o processNode.o serverNode.o

all: $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(OBJECTS) -o hydrodynamics

%.o: %.cpp
	$(MPI_CC) $(CFLAGS) $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) $< -o $@

clean:
	rm *.o
