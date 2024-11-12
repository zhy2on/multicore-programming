#pragma warning(disable : 4996)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <windows.h>
#include <math.h>
#include <time.h>
#include <direct.h>

extern const char* CLASS_NAME[];


static void convolution(float* inputs, float* outputs, float* filter, float* biases, int inDim, int outDim, int nbyn) {

	memset(outputs, 0, nbyn * nbyn * outDim * sizeof(float));
	int x = 0, y = 0;
	int offset = nbyn * nbyn;
	float sum = 0, temp;
	float* input, * output;
	
	for (int outNeuron = 0; outNeuron < outDim; ++outNeuron) {
		input = inputs;
		for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
			output = outputs;
			for (int row = 0; row < nbyn; ++row) {
				for (int col = 0; col < nbyn; ++col) {
					sum = 0;
					for (int fRow = 0; fRow < 3; ++fRow) {
						for (int fCol = 0; fCol < 3; ++fCol) {
							x = col + fCol - 1;
							y = row + fRow - 1;

							if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
								sum += input[nbyn * y + x] * filter[3 * fRow + fCol];
							}

						}
					}
					*(output++) += sum;
				}
			}
			filter += 9;
			input += offset;

		}
		for (int i = 0; i < offset; ++i) {
			(*outputs) = (*outputs) + (*biases);
			if (*outputs < 0) (*outputs) = 0;	//ReLU
			outputs++;
		}
		++biases;
	}

}

static void max_pooling(float* input, float* output, int DIM, int nbyn) {
	float max,temp;
	int n, row, col, x, y;
	for (n = 0; n < DIM; ++n) {
		for (row = 0; row < nbyn; row += 2) {
			for (col = 0; col < nbyn; col += 2) {
				//max = -FLT_MAX;
				max = 0;
				for (y = 0; y < 2; ++y) {
					for (x = 0; x < 2; ++x) {
						temp = input[nbyn * (row + y) + col + x];
						if (max < temp) max = temp;
					}
				}
				*(output++) = max;
			}
		}
		input += nbyn * nbyn;
	}
}


void fc_layer(float* input, float* output, float* weights, float* biases, int inDim, int outDim) {
	float sum;
	for (int outNeuron = 0; outNeuron < outDim; ++outNeuron) {
		sum = 0;
		for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
			sum += input[inNeuron] * (*weights++);
		}
		sum += biases[outNeuron];
		if (sum > 0) output[outNeuron] = sum;	//ReLU
		else output[outNeuron] = 0;
	}
}

static void softmax(float* input, int N) {
	int i;
	float max = input[0];
	for (i = 1; i < N; i++) {
		if (max < input[i]) max = input[i];
	}
	float sum = 0;
	for (i = 0; i < N; i++) {
		sum += exp(input[i] - max);
	}
	for (i = 0; i < N; i++) {
		input[i] = exp(input[i] - max) / (sum+1e-7);
	}
}

static int find_max(float* input, int classNum) {
	int i;
	int maxIndex = 0;
	float max = 0;
	for (i = 0; i < classNum; i++) {
		if (max < input[i]) {
			max = input[i];
			maxIndex = i;
		}
	}
	return maxIndex;
}


const int INPUT_DIM[] = {
	3, 64,
	64,

	64,128,
	128,

	128, 256, 256,
	256,

	256, 512, 512,
	512,

	512, 512, 512,
	512,

	512,
	512,
	512
};

const int OUTPUT_DIM[] = {
	64, 64,
	64,

	128, 128,
	128,

	256, 256, 256,
	256,

	512, 512, 512,
	512,

	512, 512, 512,
	512,

	512,
	512,
	10
};

const int NBYN[] = {
	32, 32,
	16,

	16, 16,
	8,

	8, 8, 8,
	4,

	4, 4, 4,
	2,

	2, 2, 2,
	1,

	1,
	1,
	1
};



void cnn_seq(float* images, float* network, int* labels, float* confidences, int num_of_image) {
	
	float* w[21];
	float* b[21];
	int offset = 0;
	// link weights and biases to network
	for (int i = 0; i < 17; ++i) {
		if (i == 2 || i == 5 || i == 9 || i == 13) i++;	// pooling layer has no weights and biases
		w[i] = network + offset;
		offset += 3 * 3 * INPUT_DIM[i] * OUTPUT_DIM[i];
		b[i] = network + offset;
		offset += OUTPUT_DIM[i];
	}
	for (int i = 18; i < 21; ++i) {
		w[i] = network + offset;
		offset += INPUT_DIM[i] * OUTPUT_DIM[i];
		b[i] = network + offset;
		offset += OUTPUT_DIM[i];
	}


	// allocate memory for layer
	float* layer[21];
	for (int i = 0; i < 21; ++i) {
		layer[i] = (float*)malloc(sizeof(float) * OUTPUT_DIM[i] * NBYN[i] * NBYN[i]);
		if (layer[i] == NULL) {
			perror("malloc error");
		}
	}

	time_t start, end;
	start = clock();
	// run network
	for (int i = 0; i < num_of_image; ++i) {
		convolution(images, layer[0], w[0], b[0], INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0]);
		convolution(layer[0], layer[1], w[1], b[1], INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1]);
		max_pooling(layer[1], layer[2], INPUT_DIM[2], NBYN[2] * 2);

		convolution(layer[2], layer[3], w[3], b[3], INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3]);
		convolution(layer[3], layer[4], w[4], b[4], INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4]);
		max_pooling(layer[4], layer[5], INPUT_DIM[5], NBYN[5] * 2);

		convolution(layer[5], layer[6], w[6], b[6], INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6]);
		convolution(layer[6], layer[7], w[7], b[7], INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7]);
		convolution(layer[7], layer[8], w[8], b[8], INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8]);
		max_pooling(layer[8], layer[9], INPUT_DIM[9], NBYN[9] * 2);

		convolution(layer[9], layer[10], w[10], b[10], INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10]);
		convolution(layer[10], layer[11], w[11], b[11], INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11]);
		convolution(layer[11], layer[12], w[12], b[12], INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12]);
		max_pooling(layer[12], layer[13], INPUT_DIM[13], NBYN[13] * 2);

		convolution(layer[13], layer[14], w[14], b[14], INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14]);
		convolution(layer[14], layer[15], w[15], b[15], INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15]);
		convolution(layer[15], layer[16], w[16], b[16], INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16]);
		max_pooling(layer[16], layer[17], INPUT_DIM[17], NBYN[17] * 2);

		fc_layer(layer[17], layer[18], w[18], b[18], INPUT_DIM[18], OUTPUT_DIM[18]);
		fc_layer(layer[18], layer[19], w[19], b[19], INPUT_DIM[19], OUTPUT_DIM[19]);
		fc_layer(layer[19], layer[20], w[20], b[20], INPUT_DIM[20], OUTPUT_DIM[20]);

		softmax(layer[20], 10);

		labels[i] = find_max(layer[20], 10);
		confidences[i] = layer[20][labels[i]];
		images += 32 * 32 * 3;
	}
	end = clock();
	printf("Elapsed time: %.2f sec\n", (double)(end - start) / CLK_TCK);

	for (int i = 0; i < 21; ++i) {
		free(layer[i]);
	}
}

