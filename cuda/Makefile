
all: c global share

c: mat_c.cu
	nvcc -o c mat_c.cu

global : mat_global.cu
	nvcc -o global mat_global.cu

share : mat_share.cu
	nvcc -o share mat_share.cu


clean:
	rm -f c global share
