#include "NN.h"

/*
TODO:
- allow custom sizes for layers and allow custom layers
- allow layers to use other layers
*/

int main()
{
	const float LEARNING_RATE = 0.00001f;
	const int BATCH_SIZE = 32;
	const int EPISODES = 10000;

	float UPDATE_RATE = LEARNING_RATE / BATCH_SIZE;
	
	NN nn(4);

	for (int episode = 0; episode < EPISODES; episode++)
	{
		for (int batch = 0; batch < BATCH_SIZE; batch++)
		{
			uint8_t a = rand();
			uint8_t b = rand();
			uint8_t c = a + b;
			/*for (int i = 0; i < ResidualLinearReluLayer::size; i++)
				nn.GetInputTensor()[i] = i + 1;*/
			for (int i = 0; i < 8; i++)
				nn.GetInputTensor()[i] = (a >> i) & 1;
			for (int i = 0; i < 8; i++)
				nn.GetInputTensor()[i + 8] = (b >> i) & 1;
			nn.Forward();

			for (int i = 0; i < ResidualLinearReluLayer::size; i++)
				nn.GetOutputGradientTensor()[i] = 7;/**/
			/*for (int i = 0; i < 8; i++)
				nn.GetOutputGradientTensor()[i] = ((c >> i) & 1) + 10;
			for (int i = 0; i < 8; i++)
				nn.GetOutputGradientTensor()[i + 8] = 10;*/
			nn.Backward();
		}
		nn.Update(&UPDATE_RATE);
	}
	/*nn.PrintForward();
	nn.PrintBackward();*/
	PrintMatrixf32(nn.GetInputTensor(), 1, ResidualLinearReluLayer::size, "Input");
	PrintMatrixf32(nn.GetOutputTensor(), 1, ResidualLinearReluLayer::size, "Output");
	PrintMatrixf32(nn.GetOutputGradientTensor(), 1, ResidualLinearReluLayer::size, "Output Gradient");

	return 0;
}