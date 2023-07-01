#pragma once
#include <iostream>
#include <vector>

const float ONEF = 1.0f;

void cpuSgemmStridedBatched(
	bool transB, bool transA,
	int CCols, int CRows, int AColsBRows,
	const float* alpha,
	float* B, int ColsB, int SizeB,
	float* A, int ColsA, int SizeA,
	const float* beta,
	float* C, int ColsC, int SizeC,
	int batchCount)
{
	for (int b = batchCount; b--;)
	{
		for (int m = CCols; m--;)
			for (int n = CRows; n--;)
			{
				float sum = 0;
				for (int k = AColsBRows; k--;)
					sum += (transA ? A[k * ColsA + n] : A[n * ColsA + k]) * (transB ? B[m * ColsB + k] : B[k * ColsB + m]);
				C[n * ColsC + m] = *alpha * sum + *beta * C[n * ColsC + m];
			}
		A += SizeA;
		B += SizeB;
		C += SizeC;
	}
}

void cpuRelu(float* data, int size)
{
	for (int i = 0; i < size; i++)
		data[i] = data[i] > 0 ? data[i] : 0;
}

void cpuReluGradient(float* gradient, float* data, int size)
{
	for (int i = 0; i < size; i++)
		gradient[i] = data[i] > 0 ? gradient[i] : 0;
}

void cpuSaxpy(int n, float a, float* x, float* y)
{
	for (int i = 0; i < n; i++)
		y[i] = a * x[i] + y[i];
}

void PrintMatrixf32(float* arr, uint32_t rows, uint32_t cols, const char* label)
{
	printf("%s:\n", label);
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < cols; j++)
			printf("%8.3f ", arr[i * cols + j]);
		printf("\n");
	}
	printf("\n");
}

uint8_t rand_uint8_t()
{
	return rand() & 0xFF;
}

void print_uint8_t(uint8_t input)
{
	for (int i = 8; i--;)
		printf("%d", (input >> i) & 1);
	printf("\n");
}