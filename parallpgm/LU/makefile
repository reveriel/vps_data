EXE = c omp

all: ${EXE}

omp: lu_mp.c config.h
	gcc -fopenmp lu_mp.c -o omp -O2 -std=c99

c: lu_c.c config.h
	gcc lu_c.c -o c -O2 -std=c99



clean:
	rm -f ${EXE}
