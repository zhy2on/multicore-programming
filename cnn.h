#ifndef _CNN_H
#define _CNN_H
void cnn_seq(float* images, float* network, int* labels, float* confidences, int num_of_image);
void compare(const char* filename, int num_of_image);
#endif 