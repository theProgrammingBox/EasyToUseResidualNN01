#pragma once
#include "Layer.h"

struct NN
{
	int inputSize;
	
	std::vector<Layer*> layers;

	NN(int inputSize)
		: inputSize(inputSize)
	{
	}

	~NN()
	{
		for (Layer* layer : layers)
			delete layer;
	}

	void AddLayer(Layer* layer)
	{
		layers.emplace_back(layer);
	}

	void Forward(const float* inputTensor)
	{
		for (Layer* layer : layers)
			layer->ZeroForward();
		
		layers.front()->Forward(inputTensor);
		for (int i = 1; i < layers.size(); ++i)
			layers[i]->Forward(layers[i - 1]->outputTensor);
	}

	void Backward(float* outputGradientTensor, const float* inputTensor)
	{
		cpuSaxpy(layers.back()->outputSize, &MINUS_ONEF, layers.back()->outputTensor, 1, outputGradientTensor, 1);
		
		for (Layer* layer : layers)
			layer->ZeroBackward();

		if (layers.size() >= 2)
		{
			layers.back()->Backward(outputGradientTensor, layers[layers.size() - 2]->outputTensor);
			for (int i = layers.size() - 2; i > 0; --i)
				layers[i]->Backward(layers[i + 1]->inputGradientTensor, layers[i - 1]->outputTensor);
			layers.front()->Backward(layers[1]->inputGradientTensor, inputTensor);
		}
		else
		{
			layers.front()->Backward(outputGradientTensor, inputTensor);
		}
	}

	void Update(float* learningRate)
	{
		for (Layer* layer : layers)
			layer->Update(learningRate);
	}

	void PrintForward(float* inputTensor)
	{
		PrintMatrixf32(inputTensor, 1, inputSize, "Input Tensor");
		for (Layer* layer : layers)
			layer->PrintForward();
	}

	void PrintBackward(float* outputGradientTensor)
	{
		PrintMatrixf32(outputGradientTensor, 1, layers.back()->outputSize, "Output Gradient Tensor");
		for (Layer* layer : layers)
			layer->PrintBackward();
	}

	void PrintParams()
	{
		for (Layer* layer : layers)
			layer->PrintParams();
	}
};