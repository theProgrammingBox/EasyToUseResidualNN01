#pragma once
#include "Layer.h"

struct ResidualLinearReluLayer : Layer
{
	float* weightTensor;
	float* biasTensor;
	float* productTensor;
	
	float* productGradientTensor;
	float* weightGradientTensor;
	float* biasGradientTensor;

	ResidualLinearReluLayer(int inputSize)
		: Layer(inputSize, inputSize)
	{
		weightTensor = new float[inputSize];
		biasTensor = new float[inputSize];
		productTensor = new float[inputSize];

		productGradientTensor = new float[inputSize];
		weightGradientTensor = new float[inputSize];
		biasGradientTensor = new float[inputSize];
	}

	virtual ~ResidualLinearReluLayer() override
	{
		delete[] weightTensor;
		delete[] biasTensor;
		delete[] productTensor;

		delete[] productGradientTensor;
		delete[] weightGradientTensor;
		delete[] biasGradientTensor;
	}

	void InitParams()
	{
		memset(weightTensor, 0, sizeof(float) * inputSize * inputSize);
		memset(biasTensor, 0, sizeof(float) * inputSize);
		
		memset(weightGradientTensor, 0, sizeof(float) * inputSize * inputSize);
		memset(biasGradientTensor, 0, sizeof(float) * inputSize);
	}

	virtual void ZeroForward() override
	{
		memset(productTensor, 0, sizeof(float) * inputSize);
		memset(outputTensor, 0, sizeof(float) * inputSize);
	}

	virtual void ZeroBackward() override
	{
		memset(productGradientTensor, 0, sizeof(float) * inputSize);
		memset(inputGradientTensor, 0, sizeof(float) * inputSize);
	}

	virtual void Forward(const float* inputTensor) override
	{
		cpuSgemmStridedBatched(
			false, false,
			inputSize, 1, inputSize,
			&ONEF,
			weightTensor, inputSize, inputSize * inputSize,
			inputTensor, inputSize, inputSize,
			&ONEF,
			productTensor, inputSize, inputSize,
			1);
		cpuSaxpy(inputSize, &ONEF, biasTensor, 1, productTensor, 1);

		// residual
		cpuSaxpy(inputSize, &ONEF, inputTensor, 1, outputTensor, 1);
		cpuReluForward(inputSize, &ONEF, productTensor, &ONEF, outputTensor);
	}

	virtual void Backward(const float* outputTensor, const float* inputTensor) override
	{
		// residual
		cpuSaxpy(inputSize, &ONEF, outputTensor, 1, inputGradientTensor, 1);
		cpuReluBackward(inputSize, &ONEF, outputTensor, productTensor, &ONEF, productGradientTensor);

		cpuSaxpy(inputSize, &ONEF, productGradientTensor, 1, biasGradientTensor, 1);
		cpuSgemmStridedBatched(
			false, true,
			inputSize, inputSize, 1,
			&ONEF,
			productGradientTensor, inputSize, inputSize,
			inputTensor, inputSize, inputSize,
			&ONEF,
			weightGradientTensor, inputSize, inputSize * inputSize,
			1);
		cpuSgemmStridedBatched(
			true, false,
			inputSize, 1, inputSize,
			&ONEF,
			weightTensor, inputSize, inputSize * inputSize,
			productGradientTensor, inputSize, inputSize,
			&ONEF,
			inputGradientTensor, inputSize, inputSize,
			1);
	}

	virtual void Update(const float* learningRate) override
	{
		cpuSaxpy(inputSize, learningRate, biasGradientTensor, 1, biasTensor, 1);
		cpuSaxpy(inputSize * inputSize, learningRate, weightGradientTensor, 1, weightTensor, 1);

		memset(weightGradientTensor, 0, sizeof(float) * inputSize * inputSize);
		memset(biasGradientTensor, 0, sizeof(float) * inputSize);
	}

	virtual void PrintForward() const override
	{
		PrintMatrixf32(weightTensor, inputSize, inputSize, "Weight Tensor");
		PrintMatrixf32(biasTensor, 1, inputSize, "Bias Tensor");
		PrintMatrixf32(productTensor, 1, inputSize, "Product Tensor");
		PrintMatrixf32(outputTensor, 1, inputSize, "Residual Sum Tensor");
		printf("\n");
	}

	virtual void PrintBackward() const override
	{
		PrintMatrixf32(productGradientTensor, 1, inputSize, "Product Gradient Tensor");
		PrintMatrixf32(weightGradientTensor, inputSize, inputSize, "Weight Gradient Tensor");
		PrintMatrixf32(biasGradientTensor, 1, inputSize, "Bias Gradient Tensor");
		PrintMatrixf32(inputGradientTensor, 1, inputSize, "Input Gradient Tensor");
		printf("\n");
	}

	virtual void PrintParams() const override
	{
		PrintMatrixf32(weightTensor, inputSize, inputSize, "Weight Tensor");
		PrintMatrixf32(biasTensor, 1, inputSize, "Bias Tensor");
		printf("\n");
	}
};