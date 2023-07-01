#include "NN.h"

/*
TODO:
- allow custom sizes for layers and allow custom layers
- allow layers to use other layers
- layer norm
*/

int main()
{
	const float LEARNING_RATE = 0.002f;
	const int BATCH_SIZE = 1;
	const int EPISODES = 10;

	float UPDATE_RATE = LEARNING_RATE / BATCH_SIZE;

	const int inputSize = 8;
	float inputTensor[inputSize];
	float outputGradientTensor[inputSize];
	
	NN nn(inputSize);
	nn.AddLayer(new ResidualLinearReluLayer(inputSize));
	nn.AddLayer(new ResidualLinearReluLayer(inputSize));

	assert(nn.inputSize == inputSize);
	assert(nn.layers.back()->outputSize == inputSize);
	
	for (int episode = 0; episode < EPISODES; episode++)
	{
		for (int batch = 0; batch < BATCH_SIZE; batch++)
		{
			for (int i = 0; i < inputSize; i++)
				inputTensor[i] = i + 1;
			nn.Forward(inputTensor);

			for (int i = 0; i < inputSize; i++)
				outputGradientTensor[i] = inputTensor[(i * 5 + 3) % inputSize] + i;
			nn.Backward(outputGradientTensor, inputTensor);
		}
		nn.Update(&UPDATE_RATE);
	}
	nn.PrintForward(inputTensor);
	nn.PrintBackward(outputGradientTensor);

	return 0;
}