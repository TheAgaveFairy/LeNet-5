#include "lenet.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
//#include "model.h"
#include "lenet5_model.h"
//#include "quantweights.h"


#define FILE_TRAIN_IMAGE		"train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL		"train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE		"t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL		"t10k-labels-idx1-ubyte"
#define LENET_FILE 		"model.dat"
#define COUNT_TRAIN		60000
#define COUNT_TEST		10000

#define DEBUG 0

// Paulie D. reads in the next row of data from an already loaded file, return 1 if success
int read_from_csv(FILE *fp, int n, image img) {
	char *line = NULL;
	size_t len = 0;
	int read; // ssize_t
	int label_out = -1;
	
	if ((read = getline(&line, &len, fp)) == -1) {
		free(line);
		fprintf(stderr, "Error reading line from csv.\n");
		return -1;
	}

	//if (DEBUG) printf("Read in from csv: %s\n", line);

	char *ptr = line; // start at start of line
	label_out = atoi(ptr); // label is a single digit at the start of the line
	
	ptr = strchr(ptr, ',');
	if (!ptr) {
		free(line);
		fprintf(stderr, "File format error.\n");
		return -1;
	}
	ptr++; // hmm

	//if (DEBUG) printf("Reading line digits:\n");
	for (int r = 0; r < n; r++){
		for (int c = 0; c < n; c++) {
			uint8 found = atoi(ptr);
			img[r][c] = found;
			//if (DEBUG) printf("%d. ", found);

			char *next = strchr(ptr, ',');
			if (!next && (r * n + c) < n * n - 1) {
				free(line);
				fprintf(stderr, "Not enough digits in line found.\n");
				return -1;
			}

			if (next) ptr = next + 1;
		}
	}

	free(line);
	return label_out;
}

//Paulie D. loads in file and skips the header so we're pointing to the first actual desired line
FILE * load_csv_file(const char* filename) {
	FILE *fp = fopen(filename, "r");
	if (!fp) {
		fprintf(stderr, "Error opening csv file.\n");
		return NULL;
	}

	// skip the header (if present)!!!
	char first_char = fgetc(fp);
	ungetc(first_char, fp);
	if (first_char < '0' || first_char > '9') {
		char buffer[8192]; // needs to be as big or bigger than the header's size
		char *suppress = fgets(buffer, sizeof(buffer), fp);
	}

	return fp; // start of actual data is returned
}

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

int main()
{
	FILE *csv = load_csv_file("mnist_test-1.csv"); // header skipped
	if (!csv) {
		fprintf(stderr, "Csv not found, exiting\n");
		return 1;
	}
	
	//LeNet5Quantized *lenet = (LeNet5Quantized *) malloc(sizeof(LeNet5Quantized));
	LeNet5Quantized *lenet = &lenet5_model;
	if (!lenet) {
		fprintf(stderr, "Failed to allocate LeNet5\n");
		return 1;
	}
	loadQuantized(lenet, "quant.dat");

	int correct = 0;
	int num_to_test = 2500;
	for (int i = 0; i < num_to_test; i++) { // test 100 images
		image img;
		int test_label = read_from_csv(csv, 28, img); // returns label
		if (test_label < 0) {
			return test_label; // failure to read
		}

		//int p = Predict(lenet, img, 10); // lets go look at this
		int p = QuantPredict(lenet, img, 10); // lets go look at this
		if (p == test_label) correct++;
		if (p != test_label && DEBUG) { // && 1 to display failures
			printf("Testing digit: %d. Model predicts: %d.\n", test_label, p);
		}
		if (DEBUG) printf("Testing digit: %d. Model predicts: %d.\n", test_label, p);
	}
	//printf("Finished csv run\n");
	printf("%d/%d = %lf%% accuracy from csv testing.\n", correct, num_to_test, (double)(correct * 100.0) / num_to_test);
	
	//must include model.h to call PrintModel
	//if (DEBUG) printf("\n\n\n");
	//if (DEBUG) PrintModel(lenet);

	printf("SIZE OF FEATURE: %lu\n.", sizeof(Feature));
	return EXIT_SUCCESS;
}
