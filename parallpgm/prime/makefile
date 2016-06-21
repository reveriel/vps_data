all : c omp



c: config.h prime.c
	gcc -O2 prime.c -o c -lm -std=c99

omp : config.h prime_omp.c
	gcc -O2 prime_omp.c -o omp -fopenmp  -lm -std=c99

clean:
	rm -f c omp
