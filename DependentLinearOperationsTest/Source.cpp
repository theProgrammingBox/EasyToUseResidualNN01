#include "NN.h"

int main()
{
	NN nn(3);
	for (int i = 0; i < Layer::size; i++)
		nn.inputTensor[i] = i + 1;
	
	nn.Forward();
	nn.Print();

	return 0;
}