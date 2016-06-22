EXE = c  omp cuda

CFLAGS = -std=c99 -O2

all: ${EXE}

c: config.h mat.c
	gcc mat.c -o c  ${CFLAGS} 

omp : config.h mat_omp.c
	gcc mat_omp.c -o omp ${CFLAGS}  -fopenmp

cuda : config.h mat.cu
	nvcc mat.cu -o cuda -O2
	

clean:
	rm -f ${EXE}
