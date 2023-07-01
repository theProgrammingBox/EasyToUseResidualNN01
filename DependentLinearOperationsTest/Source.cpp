#include "NN.h"

int main()
{
	const float LEARNING_RATE = 0.0004f;
	const int BATCH_SIZE = 8;
	const int EPISODES = 1000;
	
	NN nn(2);

	for (int episode = 0; episode < EPISODES; episode++)
	{
		for (int batch = 0; batch < BATCH_SIZE; batch++)
		{
			for (int i = 0; i < Layer::size; i++)
				nn.GetInputTensor()[i] = i + 1;
			nn.Forward();

			for (int i = 0; i < Layer::size; i++)
				nn.GetOutputGradientTensor()[i] = nn.GetInputTensor()[(i + 3) % Layer::size] + i;
				//nn.GetOutputGradientTensor()[i] = nn.GetInputTensor()[i];
			nn.Backward();
		}
		nn.Update(LEARNING_RATE / BATCH_SIZE);
	}
	nn.Print();
	nn.PrintGradients();

	return 0;
}