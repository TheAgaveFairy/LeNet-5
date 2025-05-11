#include <stdio.h>

#ifndef LENGTH_FEATURE0

#define LENGTH_KERNEL 5

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

typedef struct Feature // 36,136B
{
	float input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
	float layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
	float layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
	float layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
	float layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
	float layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
	float output[OUTPUT];
}Feature;

#define HELPER(x) printf(#x " %luB\n", sizeof(x))

int main(){
	Feature feature = {0};
	HELPER(feature.input);
	HELPER(feature.layer1);
	HELPER(feature.layer2);
	HELPER(feature.layer3);
	HELPER(feature.layer4);
	HELPER(feature.layer5);
	HELPER(feature.output);

}
