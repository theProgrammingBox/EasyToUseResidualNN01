#pragma once
#include "Layer.h"
#include "ResidualLinearReluLayer.h"

struct NN
{
	std::vector<ResidualLinearReluLayer> layers;
	
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

	void Update(float* learningRate)
	{
		for (int i = 0; i < layers.size(); ++i)
			layers[i].Update(learningRate);
	}

	void Forward(float* inputTensor)
	{
		ZeroForward();
		
		layers.front().Forward(inputTensor);
		for (int i = 1; i < layers.size(); ++i)
			layers[i].Forward(layers[i - 1].GetOutputTensor());
	}

	void Backward(float* outputGradientTensor, float* inputTensor)
	{
		cpuSaxpy(ResidualLinearReluLayer::size, &MINUS_ONEF, layers.back().GetOutputTensor(), 1, outputGradientTensor, 1);
		ZeroBackward();

		layers.back().Backward(outputGradientTensor, layers[layers.size() - 2].GetOutputTensor());
		for (int i = layers.size() - 2; i > 0; --i)
			layers[i].Backward(layers[i + 1].GetInputGradientTensor(), layers[i - 1].GetOutputTensor());
		layers.front().Backward(layers[1].GetInputGradientTensor(), inputTensor);
	}

	void PrintForward(float* inputTensor)
	{
		PrintMatrixf32(inputTensor, 1, ResidualLinearReluLayer::size, "Input Tensor");
		for (int i = 0; i < layers.size(); ++i)
			layers[i].PrintForward();
	}

	void PrintBackward(float* outputGradientTensor)
	{
		PrintMatrixf32(outputGradientTensor, 1, ResidualLinearReluLayer::size, "Output Gradient Tensor");
		for (int i = 0; i < layers.size(); ++i)
			layers[i].PrintBackward();
	}

	void PrintParams()
	{
		for (int i = 0; i < layers.size(); ++i)
			layers[i].PrintParams();
	}
};