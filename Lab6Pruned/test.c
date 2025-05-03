#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// OH I MISS ZIG

#define MyType float
#define MyRand(d) (MyType)(f64rand())

static double f64rand()
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

inline int8_t quantize(float in, float s) {
	// s = scaling factor, fabs() of your range
	int8_t out = 0;
	int8_t rounded_scaled = in / s * ((1 << 7) - 1);
	return rounded_scaled;
}

int main() {
	//double highest = -INFINITY, lowest = INFINITY;
	MyType top = -INFINITY, bottom = INFINITY;

	// generate a random range
	for (int i = 0; i < 10; i++) {
		//double temp = f64rand();
		MyType test = MyRand();
		//highest = temp > highest ? temp : highest;
		//lowest = temp < lowest ? temp : lowest;
		top = test > top ? test : top;
		bottom = test < bottom ? test : bottom;
	}

	float s = fabsf(top);
	s = fabsf(bottom) > s ? fabsf(bottom) : s;

	printf("MyType: [%f, %f]\n", bottom, top);
	printf("Scaling Factor: %f\n", s);

	
	return EXIT_SUCCESS;
}
