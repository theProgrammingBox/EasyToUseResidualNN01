#pragma once
#include "ResidualLinearReluLayer.h"

struct NN
{
	std::vector<ResidualLinearReluLayer> layers;

	float inputTensor[ResidualLinearReluLayer::size];
	float outputTensor[ResidualLinearReluLayer::size];
	
	float productGradientTensor[ResidualLinearReluLayer::size];
	float inputGradientTensor[ResidualLinearReluLayer::size];
	
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
		return productGradientTensor;
	}

	float* GetInputGradientTensor()
	{
		return inputGradientTensor;
	}

	void Forward()
	{
		ZeroForward();
		
		memcpy(layers.front().GetInputTensor(), GetInputTensor(), sizeof(float) * ResidualLinearReluLayer::size);
		for (int i = 0; i < layers.size() - 1; ++i)
		{
			layers[i].Forward();
			memcpy(layers[i + 1].GetInputTensor(), layers[i].GetOutputTensor(), sizeof(float) * ResidualLinearReluLayer::size);
		}
		layers.back().Forward();
		memcpy(GetOutputTensor(), layers.back().GetOutputTensor(), sizeof(float) * ResidualLinearReluLayer::size);
	}

	void Backward()
	{
		cpuSaxpy(ResidualLinearReluLayer::size, &MINUS_ONEF, GetOutputTensor(), 1, GetOutputGradientTensor(), 1);
		ZeroBackward();
		
		memcpy(layers.back().GetOutputGradientTensor(), GetOutputGradientTensor(), sizeof(float) * ResidualLinearReluLayer::size);
		layers.back().Backward();
		for (int i = layers.size() - 2; i >= 0; --i)
		{
			memcpy(layers[i].GetOutputGradientTensor(), layers[i + 1].GetInputGradientTensor(), sizeof(float) * ResidualLinearReluLayer::size);
			layers[i].Backward();
		}
		memcpy(GetInputGradientTensor(), layers.front().GetInputGradientTensor(), sizeof(float) * ResidualLinearReluLayer::size);
	}

	void PrintForward()
	{
		for (int i = 0; i < layers.size(); ++i)
			layers[i].PrintForward();
	}

	void PrintBackward()
	{
		for (int i = 0; i < layers.size(); ++i)
			layers[i].PrintBackward();
	}

	void PrintParams()
	{
		for (int i = 0; i < layers.size(); ++i)
			layers[i].PrintParams();
	}
};