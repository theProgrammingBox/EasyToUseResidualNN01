#pragma once
#include "Header.h"

struct Layer
{
	static const int size = 8;
	
	float inputTensor[size];
	float weightTensor[size * size];
	float biasTensor[size];
	float productActivationTensor[size];
	float residualSumTensor[size];

	float residualSumGradientTensor[size];
	float productActivationGradientTensor[size];
	float weightGradientTensor[size * size];
	float biasGradientTensor[size];
	float inputGradientTensor[size];
	
	Layer()
	{
		InitParams();
	}

	void InitParams()
	{
		memset(weightTensor, 0, sizeof(float) * size * size);
		memset(biasTensor, 0, sizeof(float) * size);
		for (int i = 0; i < size; i++)
			weightTensor[i * size + i] = 1;
		
		memset(weightGradientTensor, 0, sizeof(float) * size * size);
		memset(biasGradientTensor, 0, sizeof(float) * size);
	}

	void ZeroForward()
	{
		memset(inputTensor, 0, sizeof(float) * size);
		memset(productActivationTensor, 0, sizeof(float) * size);
		memset(residualSumTensor, 0, sizeof(float) * size);
	}

	void ZeroBackward()
	{
		memset(residualSumGradientTensor, 0, sizeof(float) * size);
		memset(productActivationGradientTensor, 0, sizeof(float) * size);
		memset(inputGradientTensor, 0, sizeof(float) * size);
	}

	void Update(float learningRate)
	{
		cpuSaxpy(size, learningRate, biasGradientTensor, biasTensor);
		cpuSaxpy(size * size, learningRate, weightGradientTensor, weightTensor);
		
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
			productActivationTensor, size, size,
			1);
		cpuSaxpy(size, ONEF, biasTensor, productActivationTensor);
		cpuRelu(productActivationTensor, size);
		
		// residual
		cpuSaxpy(size, ONEF, inputTensor, residualSumTensor);
		cpuSaxpy(size, ONEF, productActivationTensor, residualSumTensor);
	}

	void Backward()
	{
		// residual
		cpuSaxpy(size, ONEF, residualSumGradientTensor, inputGradientTensor);
		cpuSaxpy(size, ONEF, residualSumGradientTensor, productActivationGradientTensor);
		
		cpuReluGradient(productActivationGradientTensor, productActivationTensor, size);
		cpuSaxpy(size, ONEF, productActivationGradientTensor, biasGradientTensor);
		cpuSgemmStridedBatched(
			false, true,
			size, size, 1,
			&ONEF,
			productActivationGradientTensor, size, size,
			inputTensor, size, size,
			&ONEF,
			weightGradientTensor, size, size * size,
			1);
		cpuSgemmStridedBatched(
			true, false,
			size, 1, size,
			&ONEF,
			weightTensor, size, size * size,
			productActivationGradientTensor, size, size,
			&ONEF,
			inputGradientTensor, size, size,
			1);
	}

	void Print()
	{
		PrintMatrixf32(inputTensor, 1, size, "Input Tensor");
		PrintMatrixf32(weightTensor, size, size, "Weight Tensor");
		PrintMatrixf32(biasTensor, 1, size, "Bias Tensor");
		PrintMatrixf32(productActivationTensor, 1, size, "Output Tensor");
		PrintMatrixf32(residualSumTensor, 1, size, "Residual Sum Tensor");
		printf("\n");
	}

	void PrintGradients()
	{
		PrintMatrixf32(residualSumGradientTensor, 1, size, "Residual Sum Gradient Tensor");
		PrintMatrixf32(productActivationGradientTensor, 1, size, "Output Gradient Tensor");
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

struct NN
{
	std::vector<Layer> layers;

	float inputTensor[Layer::size];
	float outputTensor[Layer::size];
	
	float productActivationGradientTensor[Layer::size];
	float inputGradientTensor[Layer::size];
	
	NN(int layersCount)
	{
		layers.resize(layersCount);
	}

	void ZeroForward()
	{
		for (int i = 0; i < layers.size(); ++i)
			layers[i].ZeroForward();
	}

	void ZeroBackward()
	{
		for (int i = 0; i < layers.size(); ++i)
			layers[i].ZeroBackward();
	}

	void Update(float learningRate)
	{
		for (int i = 0; i < layers.size(); ++i)
			layers[i].Update(learningRate);
	}

	float* GetInputTensor()
	{
		return inputTensor;
	}

	float* GetOutputTensor()
	{
		return outputTensor;
	}

	float* GetOutputGradientTensor()
	{
		return productActivationGradientTensor;
	}

	float* GetInputGradientTensor()
	{
		return inputGradientTensor;
	}

	void Forward()
	{
		ZeroForward();
		
		memcpy(layers.front().GetInputTensor(), GetInputTensor(), sizeof(float) * Layer::size);
		for (int i = 0; i < layers.size() - 1; ++i)
		{
			layers[i].Forward();
			memcpy(layers[i + 1].GetInputTensor(), layers[i].GetOutputTensor(), sizeof(float) * Layer::size);
		}
		layers.back().Forward();
		memcpy(GetOutputTensor(), layers.back().GetOutputTensor(), sizeof(float) * Layer::size);
	}

	void Backward()
	{
		cpuSaxpy(Layer::size, -1, GetOutputTensor(), GetOutputGradientTensor());
		ZeroBackward();
		
		memcpy(layers.back().GetOutputGradientTensor(), GetOutputGradientTensor(), sizeof(float) * Layer::size);
		layers.back().Backward();
		for (int i = layers.size() - 2; i >= 0; --i)
		{
			memcpy(layers[i].GetOutputGradientTensor(), layers[i + 1].GetInputGradientTensor(), sizeof(float) * Layer::size);
			layers[i].Backward();
		}
		memcpy(GetInputGradientTensor(), layers.front().GetInputGradientTensor(), sizeof(float) * Layer::size);
	}

	void Print()
	{
		for (int i = 0; i < layers.size(); ++i)
			layers[i].Print();
	}

	void PrintGradients()
	{
		for (int i = 0; i < layers.size(); ++i)
			layers[i].PrintGradients();
	}

	void PrintParams()
	{
		for (int i = 0; i < layers.size(); ++i)
			layers[i].PrintParams();
	}
};