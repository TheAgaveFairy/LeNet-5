#include "mnist.h"

int getNext(image img) {
	static int i = 0;
	if (i >= NUMROWS) return 0;


	for(int8_t r = 0; r < 28; r++){
		for(int8_t c = 0; c < 28; c++){
		img[r][c] = mnist[i][r * 28 + c];
	}
	return 1;
}
