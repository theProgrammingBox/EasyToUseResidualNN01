#pragma once
#include "Layer.h"

struct ResidualLinearReluLayer : Layer
{
	static const int size = 8;
	
	float weightTensor[size * size];
	float biasTensor[size];
	float productTensor[size];
	float residualSumTensor[size];
	
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
		memset(productTensor, 0, sizeof(float) * size);
		memset(residualSumTensor, 0, sizeof(float) * size);
	}

	void ZeroBackward()
	{
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

	float* GetOutputTensor()
	{
		return residualSumTensor;
	}

	float* GetInputGradientTensor()
	{
		return inputGradientTensor;
	}

	void Forward(float* inputTensor)
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

	void Backward(float* outputTensor, float* inputTensor)
	{
		// residual
		cpuSaxpy(size, &ONEF, outputTensor, 1, inputGradientTensor, 1);
		cpuReluBackward(size, &ONEF, outputTensor, productTensor, &ONEF, productGradientTensor);

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
		PrintMatrixf32(weightTensor, size, size, "Weight Tensor");
		PrintMatrixf32(biasTensor, 1, size, "Bias Tensor");
		PrintMatrixf32(productTensor, 1, size, "Product Tensor");
		PrintMatrixf32(residualSumTensor, 1, size, "Residual Sum Tensor");
		printf("\n");
	}

	void PrintBackward()
	{
		PrintMatrixf32(productGradientTensor, 1, size, "Product Gradient Tensor");
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