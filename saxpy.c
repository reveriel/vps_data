#include <stdio.h>
#include <stdlib.h>

float max(float a, float b) {
return a > b ? a : b;
}

int main(void)
{
	int N = 1<<30;
	float *x, *y;
	x = (float*)malloc(N*sizeof(float));
	y = (float*)malloc(N*sizeof(float));

	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	for (int i = 0; i < N; i++) {
		y[i] = 2.0f * x[i] + y[i];
	}



	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = max(maxError, abs(y[i]-4.0f));
	printf("Max error: %f/n", maxError);
}
