#pragma once
#include "Header.h"

struct ResidualLinearReluLayer
{
	static const int size = 8;

	float inputTensor[size];
	float weightTensor[size * size];
	float biasTensor[size];
	float productTensor[size];
	float residualSumTensor[size];

	float residualSumGradientTensor[size];
	float productGradientTensor[size];
	float weightGradientTensor[size * size];
	float biasGradientTensor[size];
	float inputGradientTensor[size];

	ResidualLinearReluLayer()
	{
		InitParams();
	}

	void InitParams()
	{
		memset(weightTensor, 0, sizeof(float) * size * size);
		memset(biasTensor, 0, sizeof(float) * size);

		memset(weightGradientTensor, 0, sizeof(float) * size * size);
		memset(biasGradientTensor, 0, sizeof(float) * size);
	}

	void ZeroForward()
	{
		memset(inputTensor, 0, sizeof(float) * size);
		memset(productTensor, 0, sizeof(float) * size);
		memset(residualSumTensor, 0, sizeof(float) * size);
	}

	void ZeroBackward()
	{
		memset(residualSumGradientTensor, 0, sizeof(float) * size);
		memset(productGradientTensor, 0, sizeof(float) * size);
		memset(inputGradientTensor, 0, sizeof(float) * size);
	}

	void Update(float* learningRate)
	{
		cpuSaxpy(size, learningRate, biasGradientTensor, 1, biasTensor, 1);
		cpuSaxpy(size * size, learningRate, weightGradientTensor, 1, weightTensor, 1);

		memset(weightGradientTensor, 0, sizeof(float) * size * size);
		memset(biasGradientTensor, 0, sizeof(float) * size);
	}

	float* GetInputTensor()
	{
		return inputTensor;
	}

	float* GetOutputTensor()
	{
		return residualSumTensor;
	}

	float* GetOutputGradientTensor()
	{
		return residualSumGradientTensor;
	}

	float* GetInputGradientTensor()
	{
		return inputGradientTensor;
	}

	void Forward()
	{
		cpuSgemmStridedBatched(
			false, false,
			size, 1, size,
			&ONEF,
			weightTensor, size, size * size,
			inputTensor, size, size,
			&ONEF,
			productTensor, size, size,
			1);
		cpuSaxpy(size, &ONEF, biasTensor, 1, productTensor, 1);

		// residual
		cpuSaxpy(size, &ONEF, inputTensor, 1, residualSumTensor, 1);
		cpuReluForward(size, &ONEF, productTensor, &ONEF, residualSumTensor);
	}

	void Backward()
	{
		// residual
		cpuSaxpy(size, &ONEF, residualSumGradientTensor, 1, inputGradientTensor, 1);
		cpuReluBackward(size, &ONEF, residualSumGradientTensor, productTensor, &ONEF, productGradientTensor);

		cpuSaxpy(size, &ONEF, productGradientTensor, 1, biasGradientTensor, 1);
		cpuSgemmStridedBatched(
			false, true,
			size, size, 1,
			&ONEF,
			productGradientTensor, size, size,
			inputTensor, size, size,
			&ONEF,
			weightGradientTensor, size, size * size,
			1);
		cpuSgemmStridedBatched(
			true, false,
			size, 1, size,
			&ONEF,
			weightTensor, size, size * size,
			productGradientTensor, size, size,
			&ONEF,
			inputGradientTensor, size, size,
			1);
	}

	void PrintForward()
	{
		PrintMatrixf32(inputTensor, 1, size, "Input Tensor");
		PrintMatrixf32(weightTensor, size, size, "Weight Tensor");
		PrintMatrixf32(biasTensor, 1, size, "Bias Tensor");
		PrintMatrixf32(productTensor, 1, size, "Output Tensor");
		PrintMatrixf32(residualSumTensor, 1, size, "Residual Sum Tensor");
		printf("\n");
	}

	void PrintBackward()
	{
		PrintMatrixf32(residualSumGradientTensor, 1, size, "Residual Sum Gradient Tensor");
		PrintMatrixf32(productGradientTensor, 1, size, "Output Gradient Tensor");
		PrintMatrixf32(weightGradientTensor, size, size, "Weight Gradient Tensor");
		PrintMatrixf32(biasGradientTensor, 1, size, "Bias Gradient Tensor");
		PrintMatrixf32(inputGradientTensor, 1, size, "Input Gradient Tensor");
		printf("\n");
	}

	void PrintParams()
	{
		PrintMatrixf32(weightTensor, size, size, "Weight Tensor");
		PrintMatrixf32(biasTensor, 1, size, "Bias Tensor");
		printf("\n");
	}
};