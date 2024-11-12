#pragma warning(disable:4996)
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

extern const char* CLASS_NAME[];

void compare(const char* filename, int num_of_image) {
	FILE* fp1, * fp2;
	int retv;
	int* correctClass, * yourClass;
	float* correctConf, * yourConf;

	correctClass = (int*)malloc(sizeof(int) * num_of_image);
	yourClass = (int*)malloc(sizeof(int) * num_of_image);
	correctConf = (float*)malloc(sizeof(float) * num_of_image);
	yourConf = (float*)malloc(sizeof(float) * num_of_image);

	fp1 = fopen("right_answer.txt", "r");
	if (fp1 == NULL) {
		perror("error while open right_answer.txt");
		exit(1);
	}
	fp2 = fopen(filename, "r");
	if (fp2 == NULL) {
		perror("error while openeing");
		exit(1);
	}
	for (int i = 0; i < num_of_image; ++i) {

		retv = fscanf(fp1, "Image %*4d : %d : %*10s\t%f\n", correctClass + i, correctConf + i);
		if (retv = 0) {
			perror("error while fscanf");
		}
		retv = fscanf(fp2, "Image %*4d : %d : %*10s\t%f\n", yourClass + i, yourConf + i);
		if (retv = 0) {
			perror("error while fscanf");
		}
		//printf("%d : %d, %f %f\n", correctClass[i], yourClass[i], correctConf[i], yourConf[i]);
		if (correctClass[i] != yourClass[i] || fabs(correctConf[i] - yourConf[i]) > 0.01) {
			printf("Images %04d\n", i);
			printf("%10s : %f is correct. but your answer is\n", CLASS_NAME[correctClass[i]], correctConf[i]);
			printf("%10s : %f\n", CLASS_NAME[yourClass[i]], yourConf[i]);
			exit(1);
		}
	}
	printf("Good");

	free(correctClass);
	free(yourClass);
	free(correctConf);
	free(yourConf);

}