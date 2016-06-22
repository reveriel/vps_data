EXE = c omp

all: ${EXE}

omp: lu_mp.c config.h
	clang-omp -fopenmp lu_mp.c -o omp -O2

c: lu_c.c config.h
	clang lu_c.c -o c -O2



clean:
	rm -f ${EXE}
