#pragma once

#include <stdint.h>
#include <stddef.h>
#include "lenet5_model.h"

#define LENGTH_KERNEL	5

#ifndef LENGTH_FEATURE0

#define LENGTH_FEATURE0	32
#define LENGTH_FEATURE1	(LENGTH_FEATURE0 - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE2	(LENGTH_FEATURE1 >> 1)
#define LENGTH_FEATURE3	(LENGTH_FEATURE2 - LENGTH_KERNEL + 1)
#define	LENGTH_FEATURE4	(LENGTH_FEATURE3 >> 1)
#define LENGTH_FEATURE5	(LENGTH_FEATURE4 - LENGTH_KERNEL + 1)

#define INPUT			1
#define LAYER1			6
#define LAYER2			6
#define LAYER3			16
#define LAYER4			16
#define LAYER5			120
#define OUTPUT          10

#endif

#define ALPHA 0.5
#define PADDING 2

typedef unsigned char uint8;
typedef uint8 image[28][28];


typedef struct Feature // 36136B
{
	float input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
	float layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
	float layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
	float layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
	float layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
	float layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
	float output[OUTPUT];
}Feature;

uint8 QuantPredict(LeNet5Quantized *model, image input, uint8 count);

