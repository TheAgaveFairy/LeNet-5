//#include "quantweights.h"
#include "lenet.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <float.h>

#pragma diag_suppress=1544 // loop counting up
#pragma diag_suppress=1545 // int vs unsigned int array access
#pragma diag_suppress=2553 // int vs unsigned int array access
#pragma diag_suppress=1531 // floating point ops in FRAM are bad
#pragma diag_suppress=1530 // divide in FRAM is bad - image normalization could be moved to RAM

#define DEBUG 0

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

#define CONVOLUTION_FORWARD(input,output,weight,bias,action)					\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			CONVOLUTE_VALID(input[x], output[y], weight[x][y]);					\
	FOREACH(j, GETLENGTH(output))												\
		FOREACH(i, GETCOUNT(output[j]))											\
		((float *)output[j])[i] = action(((float *)output[j])[i] + bias[j]);	\
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

#define DOT_PRODUCT_FORWARD(input,output,weight,bias,action)				\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			((float *)output)[y] += ((float *)input)[x] * weight[x][y];	\
	FOREACH(j, GETLENGTH(bias))												\
		((float *)output)[j] = action(((float *)output)[j] + bias[j]);	\
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
	return answer;
}

inline float quantizeUp(int8_t in, float s) {
	float answer = ((float)(in) / RANGELIMIT) * s; // out [-s, s]
	return answer;
}

float UnpackLayer(int8_t *in_layer, float *out_layer, float scale, size_t n) {
	int8_t *pos = in_layer;
	for (size_t i = 0; i < n; i++) {
		out_layer[i] = quantizeUp(pos[i], scale);
	}
	return scale;
}

// Define a persistent buffer in FRAM for one channel of weights
#pragma PERSISTENT(w4_5buffer)
float w4_5buffer[LAYER5][LENGTH_KERNEL][LENGTH_KERNEL] = {0};

// Convolution function for layer 4-5
void convolve_layer4_5(
    float input[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4],
    float output[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5],
    int8_t weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL],
    int8_t bias4_5[LAYER5],
    float w4_5s,
    float b4_5s,
    float (*action)(float)
) {
    // Initialize output to zeros
    memset(output, 0, sizeof(float) * LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5);
    
    // Process one input channel at a time
    for (int c = 0; c < LAYER4; c++) {
        // Unquantize weights for this channel
        for (int y = 0; y < LAYER5; y++) {
            for (int k = 0; k < LENGTH_KERNEL; k++) {
                for (int l = 0; l < LENGTH_KERNEL; l++) {
                    w4_5buffer[y][k][l] = quantizeUp(weight4_5[c][y][k][l], w4_5s);
                }
            }
        }
        
        // Perform convolution for this channel
        for (int y = 0; y < LAYER5; y++) {
            for (int i = 0; i < LENGTH_FEATURE5; i++) {
                for (int j = 0; j < LENGTH_FEATURE5; j++) {
                    for (int k = 0; k < LENGTH_KERNEL; k++) {
                        for (int l = 0; l < LENGTH_KERNEL; l++) {
                            output[y][i][j] += input[c][i+k][j+l] * w4_5buffer[y][k][l];
                        }
                    }
                }
            }
        }
    }
    
    // Apply bias and activation
    for (int y = 0; y < LAYER5; y++) {
        float bias_val = quantizeUp(bias4_5[y], b4_5s);
        for (int i = 0; i < LENGTH_FEATURE5; i++) {
            for (int j = 0; j < LENGTH_FEATURE5; j++) {
                output[y][i][j] = action(output[y][i][j] + bias_val);
            }
        }
    }
}


// ow, ob = "o"riginal weights, biases
// uw, ub = "u"npacked versions
// ws, bs = "s"caling factors
#define HELPER(ow, ob, uw, ub, ws, bs)										\
	num_weights = sizeof(ow) / sizeof(int8_t);								\
	num_bias = sizeof(ob) / sizeof(int8_t);									\
	UnpackLayer((int8_t *)&ow, (float *)uw, ws, num_weights);				\
	UnpackLayer((int8_t *)&ob, (float *)ub, bs, num_bias)

static void QuantForward(LeNet5Quantized *model, Feature *features, float(*action)(float)) {
	size_t num_weights, num_bias; // used by HELPER macro

	{
		float w0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL] = {0};
		float b0_1[LAYER1] = {0};

		/* METHOD OF USING FLAT CONST MODEL
		int8_t w[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL] = {0};
		int8_t b[LAYER1] = {0};
		int8_t *wpos = lenet;
		void *suppress = memcpy(w, wpos, model_helper.sz_w0_1); //dest src bytes
		int8_t *bpos = lenet + offset_bias;
		suppress = memcpy(b, bpos, model_helper.sz_b0_1);
		
		HELPER(w, b, w0_1, b0_1, wscales[0], bscales[0]);
		CONVOLUTION_FORWARD(features->input, features->layer1, w0_1, b0_1, action);
		*/
		HELPER(model->weight0_1, model->bias0_1, w0_1, b0_1, model->w0_1s, model->b0_1s);
		CONVOLUTION_FORWARD(features->input, features->layer1, w0_1, b0_1, action);


	}

	
	SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);

	{
		float w2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
		float b2_3[LAYER3];

		HELPER(model->weight2_3, model->bias2_3, w2_3, b2_3, model->w2_3s, model->b2_3s);
		CONVOLUTION_FORWARD(features->layer2, features->layer3, w2_3, b2_3, action);
	}

	SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
	
	{
		convolve_layer4_5(
        	features->layer4,
        	features->layer5,
        	model->weight4_5,
        	model->bias4_5,
        	model->w4_5s,
        	model->b4_5s,
        	action
    	);

		
			//HELPER(model->weight4_5, model->bias4_5, w4_5, b4_5, model->w4_5s, model->b4_5s);
			//CONVOLUTION_FORWARD(features->layer4, features->layer5, w4_5, b4_5, action);
		
	}

	{
		float w5_6[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT];
		float b5_6[OUTPUT];

		HELPER(model->weight5_6, model->bias5_6, w5_6, b5_6, model->w5_6s, model->b5_6s);
		DOT_PRODUCT_FORWARD(features->layer5, features->output, w5_6, b5_6, action);
	}
}

static inline void load_input(Feature *features, const image input)
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

static uint8 get_result(Feature *features, uint8 count)
{
	float *output = (float *)features->output; 
	//const int outlen = GETCOUNT(features->output); // should equal arg count - PD
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

#pragma PERSISTENT(features)
Feature features = {0};

uint8 QuantPredict(LeNet5Quantized *model, const image input, uint8 count) {
	load_input(&features, input);
	QuantForward(model, &features, relu);
	return get_result(&features, count);
}
