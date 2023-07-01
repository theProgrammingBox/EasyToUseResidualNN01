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
	float outputTensor[size];

	float outputGradientTensor[size];
	float weightGradientTensor[size * size];
	float biasGradientTensor[size];
	float inputGradientTensor[size];
	
	Layer()
	{
		memset(inputTensor, 0, sizeof(float) * size);
		memset(weightTensor, 0, sizeof(float) * size * size);
		memset(biasTensor, 0, sizeof(float) * size);
		memset(outputTensor, 0, sizeof(float) * size);

		// set weight tensor to identity matrix
		for (int i = 0; i < size; i++)
			weightTensor[i * size + i] = 1;

		memset(outputGradientTensor, 0, sizeof(float) * size);
		memset(weightGradientTensor, 0, sizeof(float) * size * size);
		memset(biasGradientTensor, 0, sizeof(float) * size);
		memset(inputGradientTensor, 0, sizeof(float) * size);
	}

	void Forward()
	{
		cpuSgemmStridedBatched(
			false, false,
			size, 1, size,
			&alpha,
			weightTensor, size, size * size,
			inputTensor, size, size,
			&beta,
			outputTensor, size, size,
			1);
		cpuSaxpy(size, 1, biasTensor, outputTensor);
		cpuRelu(outputTensor, size);
	}

	void Print()
	{
		PrintMatrixf32(inputTensor, 1, size, "Input Tensor");
		PrintMatrixf32(weightTensor, size, size, "Weight Tensor");
		PrintMatrixf32(biasTensor, 1, size, "Bias Tensor");
		PrintMatrixf32(outputTensor, 1, size, "Output Tensor");
	}
};

const float Layer::alpha = 1;
const float Layer::beta = 0;

struct NN
{
	std::vector<Layer> layers;

	float inputTensor[Layer::size];
	float outputTensor[Layer::size];
	
	NN(int layersCount)
	{
		layers.resize(layersCount);
	}

	void Forward()
	{
		memcpy(layers[0].inputTensor, inputTensor, sizeof(float) * Layer::size);
		for (int i = 0; i < layers.size() - 1; ++i)
		{
			layers[i].Forward();
			
			// residual
			memcpy(layers[i + 1].inputTensor, layers[i].inputTensor, sizeof(float) * Layer::size);
			cpuSaxpy(Layer::size, 1, layers[i].outputTensor, layers[i + 1].inputTensor);
		}

		layers[layers.size() - 1].Forward();

		// residual
		memcpy(outputTensor, layers[layers.size() - 1].inputTensor, sizeof(float) * Layer::size);
		cpuSaxpy(Layer::size, 1, layers[layers.size() - 1].outputTensor, outputTensor);
		
	}

	void Print()
	{
		PrintMatrixf32(inputTensor, 1, Layer::size, "Input Tensor");
		for (int i = 0; i < layers.size(); ++i)
			layers[i].Print();
		PrintMatrixf32(outputTensor, 1, Layer::size, "Output Tensor");
	}
};