#include "lenet.h"
#include <stdlib.h>
#include <stdio.h>
#include <msp430.h>
//#include <time.h>
//#include <string.h>
//#include "model.h"
#include "lenet5_model.h"

#define COUNT_TEST		10000

int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image||!fp_label) return 1;
	int suppress = fseek(fp_image, 16, SEEK_SET);
	suppress = fseek(fp_label, 8, SEEK_SET);
	size_t supp = fread(data, sizeof(*data)*count, 1, fp_image);
	supp = fread(label,count, 1, fp_label);
	fclose(fp_image);
	fclose(fp_label);
	return 0;
}
int loadQuantized(LeNet5Quantized *quant, char filename[])
{
	FILE *fp = fopen(filename, "rb");
	if (!fp) return 1;
	size_t suppress = fread(quant, sizeof(LeNet5Quantized), 1, fp);
	fclose(fp);
	return 0;
}

int main(void)
{
	#ifdef __MSP430_HAS_PMM__ // Disable GPIO power-on default high-impedance mode for FRAM devices
	PM5CTL0 &= ~LOCKLPM5;
	#endif
	
	LeNet5Quantized *lenet = (LeNet5Quantized *)&lenet5_model;
	FILE *csv = 0x1337;//load_csv_file("mnist_test-1.csv");//delete


	int correct = 0;
	int num_to_test = 0;
	for (int i = 0; i < num_to_test; i++) { // test 100 images
		image img;
		int test_label = read_from_csv(csv, 28, img); // returns label
		if (test_label < 0) {
			return test_label; // failure to read
		}

		int p = QuantPredict(lenet, img, 10); // lets go look at this
		if (p == test_label) correct++;
	}
	
	printf("%d/%d = %lf%% accuracy from csv testing.\n", correct, num_to_test, (double)(correct * 100.0) / num_to_test);

	return EXIT_SUCCESS;
}
