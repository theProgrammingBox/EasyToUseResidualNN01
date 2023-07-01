#include "NN.h"

int main()
{
	const float LEARNING_RATE = 0.01f;
	const int BATCH_SIZE = 8;
	const int EPISODES = 100;
	
	NN nn(1);

	for (int episode = 0; episode < EPISODES; episode++)
	{
		for (int batch = 0; batch < BATCH_SIZE; batch++)
		{
			for (int i = 0; i < Layer::size; i++)
				nn.GetInputTensor()[i] = i + 1;
			nn.Forward();

			for (int i = 0; i < Layer::size; i++)
				nn.GetOutputGradientTensor()[i] = nn.GetInputTensor()[(i + 1) % Layer::size] + 10;
				//nn.GetOutputGradientTensor()[i] = nn.GetInputTensor()[i];
			nn.Backward();
		}
		nn.Update(LEARNING_RATE / BATCH_SIZE);
		nn.PrintParams();
	}
	nn.Print();

	return 0;
}