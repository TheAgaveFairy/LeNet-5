#include "lenet.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#define FILE_TRAIN_IMAGE		"train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL		"train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE		"t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL		"t10k-labels-idx1-ubyte"
#define LENET_FILE 		"model.dat"
#define COUNT_TRAIN		60000
#define COUNT_TEST		10000

#define DEBUG 0 // Paulie D.

// Paulie D. just a prettier progress bar, saves screen space
void showProgress(int progress, int total) {
  int bar_width = 50;
  float ratio = (float)progress / total;
  int filled = (int)(bar_width * ratio);

  printf("\r[");
  for (int i = 0; i < filled; ++i)
    printf("=");
  for (int i = filled; i < bar_width; ++i)
    printf(" ");
  printf("] %d%%", (int)(ratio * 100));

  fflush(stdout);
}

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

	if (DEBUG) printf("Read in from csv: %s\n", line);

	char *ptr = line; // start at start of line
	label_out = atoi(ptr); // label is a single digit at the start of the line
	
	ptr = strchr(ptr, ',');
	if (!ptr) {
		free(line);
		fprintf(stderr, "File format error.\n");
		return -1;
	}
	ptr++; // hmm

	if (DEBUG) printf("Reading line digits:\n");
	for (int r = 0; r < n; r++){
		for (int c = 0; c < n; c++) {
			uint8 found = atoi(ptr);
			img[r][c] = found;
			if (DEBUG) printf("%d. ", found);

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
		fgets(buffer, sizeof(buffer), fp);
	}

	return fp; // start of actual data is returned
}

//Paulie D. not the mostest cutest but sometimes it's nice to have something like this for debugging
// can pass by reference to save memory (assuming compiler is dumber than it probably is) but meh its debuggin'
void print_image(image img, int n) {
	for(int r = 0; r < n; r++){
		printf("\n");
		for (int c = 0; c < n; c++) {
			uint8 pixel = img[r][c];
			switch (pixel / 32) {
				case 0:
					printf(" ");
					break;
				case 1:
					printf(".");
					break;
				case 2:
					printf("*");
					break;
				case 3:
					printf(":");
					break;
				case 4:
					printf("x");
					break;
				case 5:
					printf("V");
					break;
				case 6:
					printf("X");
					break;
				default:
					printf("#");
			}
		}
	}
	printf("\n");
}

// the rest is all unedited
// read_data needs to be replaced
int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image||!fp_label) return 1;
	fseek(fp_image, 16, SEEK_SET);
	fseek(fp_label, 8, SEEK_SET);
	fread(data, sizeof(*data)*count, 1, fp_image);
	fread(label,count, 1, fp_label);
	fclose(fp_image);
	fclose(fp_label);
	return 0;
}

void training(LeNet5 *lenet, image *train_data, uint8 *train_label, int batch_size, int total_size)
{
	printf("Training with batch size %d:\n", batch_size); // Paulie D. better printing
	for (int i = 0, percent = 0; i <= total_size - batch_size; i += batch_size)
	{
		TrainBatch(lenet, train_data + i, train_label + i, batch_size);
		if (i * 100 / total_size > percent)
			showProgress(i, total_size);
			//printf("batchsize:%d\ttrain:%2d%%\n", batch_size, percent = i * 100 / total_size);
	}
	printf("\n");
}

int testing(LeNet5 *lenet, image *test_data, uint8 *test_label,int total_size)
{
	int right = 0, percent = 0;
	printf("Testing:\n"); // Paulie D. better printing
	for (int i = 0; i < total_size; ++i)
	{
		uint8 l = test_label[i];
		int p = Predict(lenet, test_data[i], 10);
		right += l == p;
		showProgress(i, total_size);
		//if (i * 100 / total_size > percent)
			//printf("test:%2d%%\n", percent = i * 100 / total_size);
	}
	printf("\n");
	return right;
}

int save(LeNet5 *lenet, char filename[])
{
	FILE *fp = fopen(filename, "wb");
	if (!fp) return 1;
	fwrite(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}

int load(LeNet5 *lenet, char filename[])
{
	FILE *fp = fopen(filename, "rb");
	if (!fp) return 1;
	fread(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}

void foo()
{
	image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
	uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
	image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
	uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));
	if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
		free(train_data);
		free(train_label);
		//system("pause");
	}
	if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
		free(test_data);
		free(test_label);
		//system("pause");
	}


	LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
	if (load(lenet, LENET_FILE))
		Initial(lenet);
	clock_t start = clock();
	int batches[] = { 300 };
	for (int i = 0; i < sizeof(batches) / sizeof(*batches);++i)
		training(lenet, train_data, train_label, batches[i],COUNT_TRAIN);
	int right = testing(lenet, test_data, test_label, COUNT_TEST); // Paulie D. "right" = "num correct guesses"
	printf("%d/%d\n", right, COUNT_TEST);
		printf("Time: %u clock ticks\n", (unsigned)(clock() - start));
	save(lenet, LENET_FILE);
	free(lenet);
	free(train_data);
	free(train_label);
	free(test_data);
	free(test_label);
	//system("pause"); // windows thing and dumb
}

int main()
{
	FILE *csv = load_csv_file("mnist_test-1.csv"); // header skipped
	if (!csv) {
		return 1;
	}

	int correct = 0;
	int num_to_test = 1000;
	for (int i = 0; i < num_to_test; i++) { // test 100 images
		image img;
		int test_label = read_from_csv(csv, 28, img); // returns label
		if (test_label < 0) {
			return test_label; // failure to read
		}
		if (DEBUG) {
			//printf("Recieved image: %d.\n", test_label);
			print_image(img, 28);
		}

		LeNet5 *lenet = malloc(sizeof(LeNet5));
		if (!lenet) {
			fprintf(stderr, "Failed to allocate LeNet5\n");
			return 1;
		}
		load(lenet, LENET_FILE);
		int p = Predict(lenet, img, 10); // lets go look at this
		if (p != test_label && 0) { // && 1 to display failures
			printf("Testing digit: %d. Model predicts: %d.\n", test_label, p);
			print_image(img, 28);
		}
		if (DEBUG) printf("Testing digit: %d. Model predicts: %d.\n", test_label, p);
	}
	//foo();
	return 0;
}
