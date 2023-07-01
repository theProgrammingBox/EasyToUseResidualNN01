#include "NN.h"

/*
TODO:
*/

int main()
{
	const float LEARNING_RATE = 0.0004f;
	const int BATCH_SIZE = 8;
	const int EPISODES = 1000;

	float UPDATE_RATE = LEARNING_RATE / BATCH_SIZE;
	
	NN nn(2);

	for (int episode = 0; episode < EPISODES; episode++)
	{
		for (int batch = 0; batch < BATCH_SIZE; batch++)
		{
			for (int i = 0; i < Layer::size; i++)
				nn.GetInputTensor()[i] = i + 1;
			nn.Forward();

			for (int i = 0; i < Layer::size; i++)
				nn.GetOutputGradientTensor()[i] = nn.GetInputTensor()[(i * 5 + 3) % Layer::size] + i;
				//nn.GetOutputGradientTensor()[i] = nn.GetInputTensor()[i];
			nn.Backward();
		}
		nn.Update(&UPDATE_RATE);
	}
	nn.PrintForward();
	nn.PrintBackward();

	return 0;
}