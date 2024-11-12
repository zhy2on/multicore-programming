// 1단계 기본 합성곱 연산을 위한 OpenCL 커널
// 각 스레드가 출력 특징 맵의 한 픽셀을 담당하여 계산
__kernel void conv_basic(
    __global float* inputs,     // 입력 특징 맵
    __global float* outputs,    // 출력 특징 맵
    __global float* filters,    // 3x3 필터 가중치
    __global float* biases,     // 편향값
    const int inDim,           // 입력 채널 수
    const int outDim,          // 출력 채널 수
    const int nbyn)            // 특징 맵의 크기 (N x N)
{
    // 현재 스레드의 좌표 계산
    const int col = get_global_id(0);    // x 좌표
    const int row = get_global_id(1);    // y 좌표
    
    // 이미지 범위를 벗어나면 종료
    if (col >= nbyn || row >= nbyn) return;
    
    // 현재는 첫 번째 출력 채널만 처리
    const int outNeuron = 0;
    float sum = 0.0f;
    
    // 모든 입력 채널에 대해 합성곱 수행
    for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
        // 3x3 필터 연산
        for (int fRow = 0; fRow < 3; ++fRow) {
            for (int fCol = 0; fCol < 3; ++fCol) {
                int y = row + fRow - 1;    // 입력 y좌표 (패딩 고려)
                int x = col + fCol - 1;    // 입력 x좌표 (패딩 고려)
                
                // 유효한 입력 좌표인 경우에만 계산
                if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
                    float input_val = inputs[inNeuron * nbyn * nbyn + y * nbyn + x];
                    float filter_val = filters[outNeuron * inDim * 9 + inNeuron * 9 + fRow * 3 + fCol];
                    sum += input_val * filter_val;
                }
            }
        }
    }
    
    // 결과 저장 (편향 더하고 ReLU 활성화 함수 적용)
    const int output_idx = outNeuron * nbyn * nbyn + row * nbyn + col;
    outputs[output_idx] = sum + biases[outNeuron];
    if (outputs[output_idx] < 0) outputs[output_idx] = 0;    // ReLU
}
