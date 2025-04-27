#include "lenet.h"
#include <memory.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <float.h>

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(float))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)

#define CONVOLUTE_VALID(input,output,weight)											\
{																						\
	FOREACH(o0,GETLENGTH(output))														\
		FOREACH(o1,GETLENGTH(*(output)))												\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];	\
}

#define CONVOLUTE_FULL(input,output,weight)												\
{																						\
	FOREACH(i0,GETLENGTH(input))														\
		FOREACH(i1,GETLENGTH(*(input)))													\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];	\
}

#define CONVOLUTION_FORWARD(input,output,weight,bias,action)					\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			CONVOLUTE_VALID(input[x], output[y], weight[x][y]);					\
	FOREACH(j, GETLENGTH(output))												\
		FOREACH(i, GETCOUNT(output[j]))											\
		((float *)output[j])[i] = action(((float *)output[j])[i] + bias[j]);	\
}

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);			\
	FOREACH(i, GETCOUNT(inerror))											\
		((float *)inerror)[i] *= actiongrad(((float *)input)[i]);			\
	FOREACH(j, GETLENGTH(outerror))											\
		FOREACH(i, GETCOUNT(outerror[j]))									\
		bd[j] += ((float *)outerror[j])[i];								\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_VALID(input[x], wd[x][y], outerror[y]);				\
}


#define SUBSAMP_MAX_FORWARD(input,output)														\
{																								\
	const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));								\
	const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));								\
	FOREACH(i, GETLENGTH(output))																\
	FOREACH(o0, GETLENGTH(*(output)))															\
	FOREACH(o1, GETLENGTH(**(output)))															\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];								\
	}																							\
}

#define SUBSAMP_MAX_BACKWARD(input,inerror,outerror)											\
{																								\
	const int len0 = GETLENGTH(*(inerror)) / GETLENGTH(*(outerror));							\
	const int len1 = GETLENGTH(**(inerror)) / GETLENGTH(**(outerror));							\
	FOREACH(i, GETLENGTH(outerror))																\
	FOREACH(o0, GETLENGTH(*(outerror)))															\
	FOREACH(o1, GETLENGTH(**(outerror)))														\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		inerror[i][o0*len0 + x0][o1*len1 + x1] = outerror[i][o0][o1];							\
	}																							\
}

#define DOT_PRODUCT_FORWARD(input,output,weight,bias,action)				\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			((float *)output)[y] += ((float *)input)[x] * weight[x][y];	\
	FOREACH(j, GETLENGTH(bias))												\
		((float *)output)[j] = action(((float *)output)[j] + bias[j]);	\
}

#define DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)	\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			((float *)inerror)[x] += ((float *)outerror)[y] * weight[x][y];	\
	FOREACH(i, GETCOUNT(inerror))												\
		((float *)inerror)[i] *= actiongrad(((float *)input)[i]);				\
	FOREACH(j, GETLENGTH(outerror))												\
		bd[j] += ((float *)outerror)[j];										\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			wd[x][y] += ((float *)input)[x] * ((float *)outerror)[y];			\
}

float relu(float x)
{
	return x*(x > 0);
}

float relugrad(float y)
{
	return y > 0;
}

float rangeAbsMax(float *arr, size_t n) {
	float highest_abs = 0.0f;
	for (size_t i = 0; i < n; i++) {
		highest_abs = fabsf(arr[i]) > highest_abs ? fabsf(arr[i]) : highest_abs;
	}
	return highest_abs;
}

#define RANGELIMIT ((1 << 7) - 1) // 127 for signed int8_t, writing this way for mathmatical clarity

inline int8_t quantizeDown(float in, float s) {
	int8_t answer = (in / s) * RANGELIMIT; // out [-127, 127]
	
	if (answer > 127.0f) { // do i need this? please compile out i pray
		fprintf(stderr, "Quantize Down out of range!\n");
		answer = 127.0f;
	}
	if (answer < -127.0f) {
		fprintf(stderr, "Quantize Down out of range!\n");
		answer = -127.0f;
	}
	return answer;
}

inline float quantizeUp(int8_t in, float s) {
	float answer = ((float)(in) / RANGELIMIT) * s; // out [-s, s]
	return answer;
}

float makeQuantizeAware(float *arr, size_t n) {
	float range_max = rangeAbsMax(arr, n);
	//printf("Range Max: %f\n", range_max);
	for (size_t i = 0; i < n; i++) {
		float old = arr[i];
		int8_t down =  quantizeDown(arr[i], range_max);
		float up = quantizeUp(down, range_max);
		arr[i] = up;
		if (i % 500 == 0 && old > 0.0f && 0) printf("RANGE: %f, old: %f, down %d, up: %f\n", range_max, old, down, up); // show some examples
	}
	return range_max; // can save as scaling factor
}


static void forward(LeNet5 *lenet, Feature *features, float(*action)(float))
{
	makeQuantizeAware((float *)features->input, GETCOUNT(features->input));//, input_size / sizeof(float));	
	CONVOLUTION_FORWARD(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action);
	
	makeQuantizeAware((float *)features->layer1, GETCOUNT(features->layer1));//, input_size / sizeof(float));	
	SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
	
	makeQuantizeAware((float *)features->layer2, GETCOUNT(features->layer2));//, input_size / sizeof(float));	
	CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action);
	
	makeQuantizeAware((float *)features->layer3, GETCOUNT(features->layer3));//, input_size / sizeof(float));	
	SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
	
	makeQuantizeAware((float *)features->layer4, GETCOUNT(features->layer4));//, input_size / sizeof(float));	
	CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action);
	
	makeQuantizeAware((float *)features->layer5, GETCOUNT(features->layer5));//, input_size / sizeof(float));	
	DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
}

static void backward(LeNet5 *lenet, LeNet5 *deltas, Feature *errors, Feature *features, float(*actiongrad)(float))
{
	DOT_PRODUCT_BACKWARD(features->layer5, errors->layer5, errors->output, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6, actiongrad);
	CONVOLUTION_BACKWARD(features->layer4, errors->layer4, errors->layer5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);
	CONVOLUTION_BACKWARD(features->layer2, errors->layer2, errors->layer3, lenet->weight2_3, deltas->weight2_3, deltas->bias2_3, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);
	CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1, lenet->weight0_1, deltas->weight0_1, deltas->bias0_1, actiongrad);
}



static inline void load_input(Feature *features, image input)
{
	float (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input;
	const long sz = sizeof(image) / sizeof(**input);
	float mean = 0.0, std = 0.0;
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(*input) / sizeof(**input))
	{
		mean += input[j][k];
		std += input[j][k] * input[j][k];
	}
	mean /= sz;
	std = sqrtf(std / sz - mean*mean);

	float highest = -INFINITY, lowest = INFINITY;

	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(*input) / sizeof(**input))
	{
		float temp = (input[j][k] - mean) / std;
		highest = temp > highest ? temp : highest;
		lowest = temp < lowest ? temp : lowest;
		layer0[0][j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
	}
		//printf("High: %lf, Low: %lf\n", highest, lowest);
}

static inline void softmax(float input[OUTPUT], float loss[OUTPUT], int label, int count)
{
	float inner = 0;
	for (int i = 0; i < count; ++i)
	{
		float res = 0;
		for (int j = 0; j < count; ++j)
		{
			res += expf(input[j] - input[i]);
		}
		loss[i] = 1. / res;
		inner -= loss[i] * loss[i];
	}
	inner += loss[label];
	for (int i = 0; i < count; ++i)
	{
		loss[i] *= (i == label) - loss[i] - inner;
	}
}

static void load_target(Feature *features, Feature *errors, int label)
{
	float *output = (float *)features->output;
	float *error = (float *)errors->output;
	softmax(output, error, label, GETCOUNT(features->output));
}

static uint8 get_result(Feature *features, uint8 count)
{
	float *output = (float *)features->output; 
	const int outlen = GETCOUNT(features->output);
	uint8 result = 0;
	float maxvalue = *output;
	for (uint8 i = 1; i < count; ++i)
	{
		if (output[i] > maxvalue)
		{
			maxvalue = output[i];
			result = i;
		}
	}
	return result;
}

static float f64rand()
{
	static int randbit = 0;
	if (!randbit)
	{
		srand((unsigned)time(0));
		for (int i = RAND_MAX; i; i >>= 1, ++randbit);
	}
	unsigned long long lvalue = 0x4000000000000000L;
	int i = 52 - randbit;
	for (; i > 0; i -= randbit)
		lvalue |= (unsigned long long)rand() << i;
	lvalue |= (unsigned long long)rand() >> -i;
	return *(double *)&lvalue - 3;
}


void TrainBatch(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize)
{
	float buffer[GETCOUNT(LeNet5)] = { 0 };
	int i = 0;
#pragma omp parallel for
	for (i = 0; i < batchSize; ++i)
	{
		Feature features = { 0 };
		Feature errors = { 0 };
		LeNet5	deltas = { 0 };
		load_input(&features, inputs[i]);
		forward(lenet, &features, relu);
		load_target(&features, &errors, labels[i]);
		backward(lenet, &deltas, &errors, &features, relugrad);
		#pragma omp critical
		{
			FOREACH(j, GETCOUNT(LeNet5))
				buffer[j] += ((float *)&deltas)[j];
		}
	}
	float k = ALPHA / batchSize;
	FOREACH(i, GETCOUNT(LeNet5))
		((float *)lenet)[i] += k * buffer[i];
}

void Train(LeNet5 *lenet, image input, uint8 label)
{
	Feature features = { 0 };
	Feature errors = { 0 };
	LeNet5 deltas = { 0 };
	load_input(&features, input);
	forward(lenet, &features, relu);
	load_target(&features, &errors, label);
	backward(lenet, &deltas, &errors, &features, relugrad);
	FOREACH(i, GETCOUNT(LeNet5))
		((float *)lenet)[i] += ALPHA * ((float *)&deltas)[i];
}

uint8 Predict(LeNet5 *lenet, image input,uint8 count)
{
	Feature features = { 0 };
	load_input(&features, input);
	forward(lenet, &features, relu);
	return get_result(&features, count);
}

float ConvertLayer(float *in_layer, int8_t *out_layer, size_t input_size) {
	//size_t input_size = sizeof(original->weight0_1);
	size_t num_elems = input_size / sizeof(float);
	float scale = rangeAbsMax(in_layer, num_elems);
	//printf("original->weight0_1 size: %luB, = %lu floats\n", input_size, num_elems);

	/*int8_t *new_quantized = (int8_t *)malloc(num_elems * sizeof(int8_t));
	if (!new_quantized) {
		fprintf(stderr, "Failure to allocate quantized layer during conversion!\n");
		return NULL;
	}*/

	float *pos = in_layer;
	for (size_t i = 0; i < num_elems; i++) {
		out_layer[i] = quantizeDown(pos[i], scale);
	}

	int printout = 1;
	if (printout) {

	pos = in_layer;
	printf("Quantized layer with scaling factor %lf:\n", scale);
	for (size_t i = 0; i < num_elems; i++) {
		printf("%d, ", out_layer[i]);
	}
	}
	return scale;
}

LeNet5Quantized * QuantizeModel(LeNet5 *original) {

	LeNet5Quantized *quantized_model = (LeNet5Quantized *)malloc(sizeof(LeNet5Quantized));
	if (!quantized_model) {
		fprintf(stderr, "Failure to allocate quantized model during conversion!\n");
		return NULL;
	}

	float scale;
	// weights
	scale = ConvertLayer((float *)&original->weight0_1, (int8_t *)quantized_model->weight0_1, sizeof(original->weight0_1));
	quantized_model->w0_1s = scale;

	scale = ConvertLayer((float *)&original->weight2_3, (int8_t *)quantized_model->weight2_3, sizeof(original->weight2_3));
	quantized_model->w2_3s = scale;

	scale = ConvertLayer((float *)&original->weight4_5, (int8_t *)quantized_model->weight4_5, sizeof(original->weight4_5));
	quantized_model->w4_5s = scale;
	
	scale = ConvertLayer((float *)&original->weight5_6, (int8_t *)quantized_model->weight5_6, sizeof(original->weight5_6));
	quantized_model->w5_6s = scale;
	
	// and biases
	scale = ConvertLayer((float *)&original->bias0_1, (int8_t *)quantized_model->bias0_1, sizeof(original->bias0_1));
	quantized_model->b0_1s = scale;
	
	scale = ConvertLayer((float *)&original->bias2_3, (int8_t *)quantized_model->bias2_3, sizeof(original->bias2_3));
	quantized_model->b2_3s = scale;
	
	scale = ConvertLayer((float *)&original->bias4_5, (int8_t *)quantized_model->bias4_5, sizeof(original->bias4_5));
	quantized_model->b4_5s = scale;
	
	scale = ConvertLayer((float *)&original->bias5_6, (int8_t *)quantized_model->bias5_6, sizeof(original->bias5_6));
	quantized_model->b5_6s = scale;

	return quantized_model;
}

void Initial(LeNet5 *lenet)
{
	for (float *pos = (float *)lenet->weight0_1; pos < (float *)lenet->bias0_1; *pos++ = (float)f64rand());
	for (float *pos = (float *)lenet->weight0_1; pos < (float *)lenet->weight2_3; *pos++ *= sqrtf(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
	for (float *pos = (float *)lenet->weight2_3; pos < (float *)lenet->weight4_5; *pos++ *= sqrtf(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
	for (float *pos = (float *)lenet->weight4_5; pos < (float *)lenet->weight5_6; *pos++ *= sqrtf(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
	for (float *pos = (float *)lenet->weight5_6; pos < (float *)lenet->bias0_1; *pos++ *= sqrtf(6.0 / (LAYER5 + OUTPUT)));
	for (int *pos = (int *)lenet->bias0_1; pos < (int *)(lenet + 1); *pos++ = 0);
}
