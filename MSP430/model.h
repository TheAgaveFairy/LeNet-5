#include <stdio.h>
#include <stdlib.h>
#include "lenet.h"

size_t PrintLayer(int8_t *layer, size_t n) {
	size_t i = 0, zeros = 0;
	for(; i < n; i++) {
		if (layer[i] == 0) zeros++;
		printf("%d, ", layer[i]);
	}
	printf("\n");
	return zeros;
}

void PrintModel(LeNet5Quantized *model) {
	#define HELPER(x) layer_size = sizeof(x);\
	num_elems = layer_size / sizeof(int8_t);\
	printf(#x " weights:\n");\
	zeros += PrintLayer((int8_t *)&x, num_elems)

	size_t layer_size, num_elems, i, zeros = 0;
	HELPER(model->weight0_1);
	HELPER(model->weight2_3);
	HELPER(model->weight4_5);
	HELPER(model->weight5_6);
	
	HELPER(model->bias0_1);
	HELPER(model->bias2_3);
	HELPER(model->bias4_5);
	HELPER(model->bias5_6);

	#define SCALAR(x) printf(#x ": %f\n", x);

	SCALAR(model->w0_1s);
	SCALAR(model->w2_3s);
	SCALAR(model->w4_5s);
	SCALAR(model->w5_6s);
	
	SCALAR(model->b0_1s);
	SCALAR(model->b2_3s);
	SCALAR(model->b4_5s);
	SCALAR(model->b5_6s);

	size_t total_elems_in_model = sizeof(LeNet5) / sizeof(float);
	double percent_zeros = (double)zeros / (double)total_elems_in_model * 100.0;
	//printf("%lu / %lu = %lf%% zeros in model.\n", zeros, total_elems_in_model, percent_zeros);
}
