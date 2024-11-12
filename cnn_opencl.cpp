#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cnn.h"

// OpenCL 관련 전역 변수
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernel;

// seq 버전의 상수 배열들을 extern으로 가져오기
extern const int INPUT_DIM[];
extern const int OUTPUT_DIM[];
extern const int NBYN[];

// 커널 소스 파일을 읽어오는 함수
char* read_kernel_source(const char* filename) {
	FILE* fp = fopen(filename, "r");
	if (fp == NULL) {
		perror("Failed to open kernel file");
		exit(1);
	}

	// 파일 크기 확인
	fseek(fp, 0, SEEK_END);
	size_t size = ftell(fp);
	rewind(fp);

	// 커널 소스를 저장할 메모리 할당
	char* source = (char*)malloc(size + 1);
	if (source == NULL) {
		perror("Failed to allocate memory for kernel source");
		exit(1);
	}

	// 파일 읽기
	fread(source, 1, size, fp);
	source[size] = '\0';
	fclose(fp);

	return source;
}

// OpenCL 초기화 함수
void cnn_init() {
	cl_int err;

	// 플랫폼 선택
	err = clGetPlatformIDs(1, &platform, NULL);
	if (err != CL_SUCCESS) {
		printf("Failed to get platform ID: %d\n", err);
		exit(1);
	}

	// GPU 디바이스 선택
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	if (err != CL_SUCCESS) {
		printf("Failed to get device ID: %d\n", err);
		exit(1);
	}

	// OpenCL 컨텍스트 생성
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Failed to create context: %d\n", err);
		exit(1);
	}

	// 커맨드 큐 생성
	queue = clCreateCommandQueue(context, device, 0, &err);
	if (err != CL_SUCCESS) {
		printf("Failed to create command queue: %d\n", err);
		exit(1);
	}

	// 커널 소스 읽기
	const char* kernel_source = read_kernel_source("conv_kernel.cl");

	// OpenCL 프로그램 생성 및 빌드
	program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Failed to create program: %d\n", err);
		exit(1);
	}

	// 프로그램 빌드
	err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		// 빌드 에러 시 로그 출력
		size_t log_size;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
							  &log_size);
		char* log = (char*)malloc(log_size);
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size,
							  log, NULL);
		printf("Build Error:\n%s\n", log);
		free(log);
		exit(1);
	}

	// 커널 생성
	kernel = clCreateKernel(program, "conv_basic", &err);
	if (err != CL_SUCCESS) {
		printf("Failed to create kernel: %d\n", err);
		exit(1);
	}

	free((void*)kernel_source);
}

// OpenCL을 사용한 합성곱 연산 함수
void convolution_cl(float* inputs, float* outputs, float* filters,
					float* biases, int inDim, int outDim, int nbyn) {
	cl_int err;

	// 입력 데이터용 버퍼 생성
	cl_mem input_buf =
		clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					   sizeof(float) * nbyn * nbyn * inDim, inputs, &err);
	cl_mem filter_buf =
		clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					   sizeof(float) * 3 * 3 * inDim * outDim, filters, &err);
	cl_mem bias_buf =
		clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					   sizeof(float) * outDim, biases, &err);
	cl_mem output_buf =
		clCreateBuffer(context, CL_MEM_WRITE_ONLY,
					   sizeof(float) * nbyn * nbyn * outDim, NULL, &err);

	// 커널 인자 설정
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buf);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buf);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &filter_buf);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &bias_buf);
	clSetKernelArg(kernel, 4, sizeof(int), &inDim);
	clSetKernelArg(kernel, 5, sizeof(int), &outDim);
	clSetKernelArg(kernel, 6, sizeof(int), &nbyn);

	// 커널 실행 설정
	size_t global_work_size[2] = {nbyn, nbyn};	// 전체 작업 크기
	size_t local_work_size[2] = {16, 16};		// 작업 그룹 크기

	// 커널 실행
	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size,
								 local_work_size, 0, NULL, NULL);

	// 결과 읽기
	err = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0,
							  sizeof(float) * nbyn * nbyn * outDim, outputs, 0,
							  NULL, NULL);

	// 버퍼 해제
	clReleaseMemObject(input_buf);
	clReleaseMemObject(filter_buf);
	clReleaseMemObject(bias_buf);
	clReleaseMemObject(output_buf);
}

// CNN 메인 함수
void cnn(float* images, float* network, int* labels, float* confidences,
		 int num_images) {
	// OpenCL 초기화
	cnn_init();

	// 가중치와 편향 포인터 배열
	float* w[21];
	float* b[21];
	int offset = 0;

	// 네트워크 파라미터 설정
	for (int i = 0; i < 17; ++i) {
		if (i == 2 || i == 5 || i == 9 || i == 13) i++;	 // 풀링 레이어 건너뛰기
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

	// 레이어별 메모리 할당
	float* layer[21];
	for (int i = 0; i < 21; ++i) {
		layer[i] =
			(float*)malloc(sizeof(float) * OUTPUT_DIM[i] * NBYN[i] * NBYN[i]);
		if (layer[i] == NULL) {
			perror("malloc error");
			return;
		}
	}

	// CNN 실행 시작
	time_t start, end;
	start = clock();

	// 이미지별 처리
	for (int i = 0; i < num_images; ++i) {
		// 첫 번째 블록 (Conv -> Conv -> Pool)
		convolution_cl(images, layer[0], w[0], b[0], INPUT_DIM[0],
					   OUTPUT_DIM[0], NBYN[0]);
		convolution_cl(layer[0], layer[1], w[1], b[1], INPUT_DIM[1],
					   OUTPUT_DIM[1], NBYN[1]);
		max_pooling(layer[1], layer[2], INPUT_DIM[2], NBYN[2] * 2);

		// ... (중간 레이어들도 동일한 패턴으로 처리)

		// 마지막 분류 레이어
		softmax(layer[20], 10);
		labels[i] = find_max(layer[20], 10);
		confidences[i] = layer[20][labels[i]];

		// 다음 이미지로
		images += 32 * 32 * 3;
	}

	end = clock();
	printf("Elapsed time: %.2f sec\n", (double)(end - start) / CLK_TCK);

	// 메모리 해제
	for (int i = 0; i < 21; ++i) {
		free(layer[i]);
	}

	// OpenCL 리소스 해제
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}
