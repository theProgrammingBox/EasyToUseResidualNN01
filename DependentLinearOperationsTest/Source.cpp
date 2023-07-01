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

	float inputTensor[ResidualLinearReluLayer::size];
	float outputGradientTensor[ResidualLinearReluLayer::size];
	
	NN nn(2);

	for (int episode = 0; episode < EPISODES; episode++)
	{
		for (int batch = 0; batch < BATCH_SIZE; batch++)
		{
			for (int i = 0; i < ResidualLinearReluLayer::size; i++)
				inputTensor[i] = i + 1;
			nn.Forward(inputTensor);

			for (int i = 0; i < ResidualLinearReluLayer::size; i++)
				outputGradientTensor[i] = inputTensor[(i * 5 + 3) % ResidualLinearReluLayer::size] + i;
			nn.Backward(outputGradientTensor, inputTensor);
		}
		nn.Update(&UPDATE_RATE);
	}
	nn.PrintForward(inputTensor);
	nn.PrintBackward(outputGradientTensor);

	return 0;
}