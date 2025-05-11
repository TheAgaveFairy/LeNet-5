#include <stdlib.h>
#include <stdio.h>
//#include <msp430.h>

#include "lenet.h"
#include "lenet5_model.h"
#include "mnist.h"

#pragma diag_suppress=1544 // loop counting up
#pragma diag_suppress=1545 // int vs unsigned int array access

//extern const LeNet5Quantized lenet5_model;

int main(void)
{
	#ifdef __MSP430_HAS_PMM__ // Disable GPIO power-on default high-impedance mode for FRAM devices
	PM5CTL0 &= ~LOCKLPM5;
	#endif
	
	LeNet5Quantized *lenet = (LeNet5Quantized *)&lenet5_model;

	int correct = 0;
	int label;
	unsigned int i = 0;
	while(i < NUMROWS) {
		int p = QuantPredict(lenet, mnist[i], 10); // lets go look at this
		label = labels[i];
		//printf("predict: %d was: %d\n", p, label);
		if (p == label) correct++;
		i++;
	}
	
	printf("%d/%d.", correct, NUMROWS);

	return EXIT_SUCCESS;
}
