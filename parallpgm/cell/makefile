EXE = c  omp cuda

CFLAGS = -std=c99 -O2

all: ${EXE}

c: config.h cell.c
	gcc cell.c -o c  ${CFLAGS} 

omp : config.h cell_omp.c
	gcc cell_omp.c -o omp ${CFLAGS}  -fopenmp

cuda : config.h cell.cu
	nvcc cell.cu -o cuda -O2
	

clean:
	rm -f ${EXE}
