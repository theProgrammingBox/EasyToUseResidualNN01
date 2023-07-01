#pragma once
#include "Header.h"

struct Layer
{
	int inputSize;
	int outputSize;
	
	float* outputTensor;
	float* inputGradientTensor;

	Layer(int inputSize, int outputSize)
	{
		this->inputSize = inputSize;
		this->outputSize = outputSize;

		outputTensor = new float[outputSize];
		inputGradientTensor = new float[inputSize];
	}
	
	virtual ~Layer()
	{
		delete[] outputTensor;
		delete[] inputGradientTensor;
	}

	virtual void ZeroForward() = 0;
	virtual void ZeroBackward() = 0;
	
	virtual void Forward(const float* inputTensor) = 0;
	virtual void Backward(const float* outputGradientTensor, const float* inputTensor) = 0;
	virtual void Update(const float* learningRate) = 0;
	
	virtual void PrintForward() const = 0;
	virtual void PrintBackward() const = 0;
	virtual void PrintParams() const = 0;
};