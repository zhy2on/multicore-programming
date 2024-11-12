#pragma warning(disable : 4996)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cnn.h"

const char* CLASS_NAME[] = {"airplane", "automobile", "bird",  "cat",  "deer",
							"dog",		"frog",		  "horse", "ship", "truck"};

void* readfile(const char* filename, int nbytes) {
	void* buf = malloc(nbytes);
	if (buf == NULL) {
		perror("error while malloc");
		exit(1);
	}

	FILE* fp = fopen(filename, "rb");
	if (fp == NULL) {
		perror("error while openeing");
		exit(1);
	}

	int retv = fread(buf, 1, nbytes, fp);
	if (retv != nbytes) {
		perror("error while read");
	}

	if (fclose(fp) != 0) {
		perror("error while closing");
		exit(1);
	}
	return buf;
}

int main(int argc, char** argv) {
	if (argc != 3) {
		perror(
			"error while get argument.\n argc[1] is (number of image)\n "
			"argc[2] is (output file name) \n");
		exit(1);
	}
	if (strcmp("right_answer.txt", argv[2]) == 0) {
		perror("'right_answer.txt' is unauthorized name");
		exit(1);
	}
	int num_of_image = atoi(argv[1]);
	if (num_of_image < 0 || num_of_image > 10000) {
		perror("number of images is 1 to 10000");
		exit(1);
	}
	float* images = (float*)readfile(
		"images.bin", sizeof(float) * 32 * 32 * 3 * num_of_image);
	float* network = (float*)readfile("network.bin", 60980520);
	int* labels = (int*)malloc(sizeof(int) * num_of_image);
	float* confidences = (float*)malloc(sizeof(float) * num_of_image);

	cnn_seq(images, network, labels, confidences, num_of_image);
	// cnn(~); //TODO
	cnn(images, network, labels, confidences, num_of_image);

	int* labels_ans = (int*)readfile("labels.bin", sizeof(int) * num_of_image);
	double acc = 0;

	FILE* fp = fopen(argv[2], "w");
	for (int i = 0; i < num_of_image; ++i) {
		fprintf(fp, "Image %04d : %d : %-10s\t%f\n", i, labels[i],
				CLASS_NAME[labels[i]], confidences[i]);
		if (labels[i] == labels_ans[i]) ++acc;
	}
	fprintf(fp, "Accuracy: %f\n", acc / num_of_image);
	fclose(fp);
	compare(argv[2], num_of_image);

	free(images);
	free(network);
	free(labels);
	free(confidences);
	free(labels_ans);

	return 0;
}
