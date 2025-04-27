/*
@author : 范文捷
@data    : 2016-04-20
@note	: 根据Yann Lecun的论文《Gradient-based Learning Applied To Document Recognition》编写
@api	:

批量训练
void TrainBatch(LeNet5 *lenet, image *inputs, const char(*resMat)[OUTPUT],uint8 *labels, int batchSize);

训练
void Train(LeNet5 *lenet, image input, const char(*resMat)[OUTPUT],uint8 label);

预测
uint8 Predict(LeNet5 *lenet, image input, const char(*resMat)[OUTPUT], uint8 count);

初始化
void Initial(LeNet5 *lenet);
*/

#pragma once

#define LENGTH_KERNEL	5

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

#define ALPHA 0.5
#define PADDING 2

typedef unsigned char uint8;
typedef uint8 image[28][28];

#define MyType float
#define MY_SQRT(x) (sizeof(x) == sizeof(float) ? sqrtf(x) : sqrt(x))
#define MY_FABS(x) (sizeof(x) == sizeof(float) ? fabsf(x) : fabs(x))
#define MY_EXP(x) (sizeof(x) == sizeof(float) ? expf(x) : exp(x))
#define MY_ZERO ((MyType)0)
#define MY_RAND(d) (MyType)(f64rand()) // not the fastest but it's fine enough

typedef struct LeNet5
{
	MyType weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
	MyType weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
	MyType weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
	MyType weight5_6[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT];

	MyType bias0_1[LAYER1];
	MyType bias2_3[LAYER3];
	MyType bias4_5[LAYER5];
	MyType bias5_6[OUTPUT];

}LeNet5;

typedef struct Feature
{
	MyType input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
	MyType layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
	MyType layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
	MyType layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
	MyType layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
	MyType layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
	MyType output[OUTPUT];
}Feature;

void TrainBatch(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize);

void Train(LeNet5 *lenet, image input, uint8 label);

uint8 Predict(LeNet5 *lenet, image input, uint8 count);

void Initial(LeNet5 *lenet);
