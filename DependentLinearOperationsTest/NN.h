#pragma once
#include "Header.h"

struct Layer
{
	static const int size = 8;
	static const float alpha;
	static const float beta;
	
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

	void ApplyGradient(float learningRate)
	{
		// sum
		// reset
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
		return productActivationGradientTensor;
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
			&alpha,
			weightTensor, size, size * size,
			GetInputTensor(), size, size,
			&beta,
			productActivationTensor, size, size,
			1);
		cpuSaxpy(size, 1, biasTensor, productActivationTensor);
		cpuRelu(productActivationTensor, size);
		
		// residual
		memcpy(GetOutputTensor(), GetInputTensor(), sizeof(float) * size);
		cpuSaxpy(size, 1, productActivationTensor, GetOutputTensor());
	}

	void Backward()
	{
		// residual
		memcpy(GetInputGradientTensor(), GetOutputGradientTensor(), sizeof(float) * size);
		memcpy(productActivationGradientTensor, GetOutputGradientTensor(), sizeof(float) * size);
		
		cpuReluGradient(productActivationGradientTensor, productActivationTensor, size);
		cpuSaxpy(size, 1, productActivationGradientTensor, biasGradientTensor);
		cpuSgemmStridedBatched(
			false, true,
			size, size, 1,
			&alpha,
			productActivationGradientTensor, size, size,
			inputTensor, size, size,
			&beta,
			weightGradientTensor, size, size * size,
			1);
		cpuSgemmStridedBatched(
			true, false,
			size, 1, size,
			&alpha,
			weightTensor, size, size * size,
			productActivationGradientTensor, size, size,
			&beta,
			inputGradientTensor, size, size,
			1);
	}

	void Print()
	{
		PrintMatrixf32(inputTensor, 1, size, "Input Tensor");
		PrintMatrixf32(weightTensor, size, size, "Weight Tensor");
		PrintMatrixf32(biasTensor, 1, size, "Bias Tensor");
		PrintMatrixf32(productActivationTensor, 1, size, "Output Tensor");
	}

	void PrintGradients()
	{
		PrintMatrixf32(productActivationGradientTensor, 1, size, "Output Gradient Tensor");
		PrintMatrixf32(weightGradientTensor, size, size, "Weight Gradient Tensor");
		PrintMatrixf32(biasGradientTensor, 1, size, "Bias Gradient Tensor");
		PrintMatrixf32(inputGradientTensor, 1, size, "Input Gradient Tensor");
	}
};

const float Layer::alpha = 1;
const float Layer::beta = 0;

struct NN
{
	std::vector<Layer> layers;

	float inputTensor[Layer::size];
	float productActivationTensor[Layer::size];
	
	float productActivationGradientTensor[Layer::size];
	float inputGradientTensor[Layer::size];
	
	NN(int layersCount)
	{
		layers.resize(layersCount);
	}

	void Forward()
	{
		memcpy(layers[0].GetInputTensor(), inputTensor, sizeof(float) * Layer::size);
		for (int i = 0; i < layers.size() - 1; ++i)
		{
			layers[i].Forward();
			memcpy(layers[i + 1].GetInputTensor(), layers[i].GetOutputTensor(), sizeof(float) * Layer::size);
		}
		layers.back().Forward();
		memcpy(productActivationTensor, layers.back().GetOutputTensor(), sizeof(float) * Layer::size);
	}

	void Backward()
	{
		cpuSaxpy(Layer::size, -1, productActivationTensor, productActivationGradientTensor);
		
		// residual
		memcpy(layers[layers.size() - 1].productActivationGradientTensor, productActivationGradientTensor, sizeof(float) * Layer::size);
		memcpy(layers[layers.size() - 1].inputGradientTensor, productActivationGradientTensor, sizeof(float) * Layer::size);
		layers[layers.size() - 1].Backward();
		
		for (int i = layers.size() - 2; i >= 0; --i)
		{
			// residual
			memcpy(layers[i].productActivationGradientTensor, layers[i + 1].inputGradientTensor, sizeof(float) * Layer::size);
			memcpy(layers[i].inputGradientTensor, layers[i + 1].inputGradientTensor, sizeof(float) * Layer::size);
			layers[i].Backward();
		}

		memcpy(inputGradientTensor, layers[0].inputGradientTensor, sizeof(float) * Layer::size);
	}

	void Print()
	{
		PrintMatrixf32(inputTensor, 1, Layer::size, "Input Tensor");
		for (int i = 0; i < layers.size(); ++i)
			layers[i].Print();
		PrintMatrixf32(productActivationTensor, 1, Layer::size, "Output Tensor");
	}

	void PrintGradients()
	{
		PrintMatrixf32(productActivationGradientTensor, 1, Layer::size, "Output Gradient Tensor");
		for (int i = 0; i < layers.size(); ++i)
			layers[i].PrintGradients();
		PrintMatrixf32(inputGradientTensor, 1, Layer::size, "Input Gradient Tensor");
	}
};