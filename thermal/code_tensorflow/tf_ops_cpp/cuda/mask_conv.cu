#include <iostream>
#include <fstream>
#include <math.h>
#include <cstdio>
#include <ctime>
#include <assert.h>     /* assert */
using namespace std;
//0 3 6 
//1 4 7
//2 5 8
// row idx: i
// col idx: j
__global__
void _mask_conv(float diag_coef_1, float diag_coef_2, float side_coef_1, float side_coef_2, int N, float *input_tensor, float *mask_tensor, float *output_tensor)
{

	float diag_coef_diff = diag_coef_1 - diag_coef_2;
	float side_coef_diff = side_coef_1 - side_coef_2;

	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;

	int stride_x = blockDim.x * gridDim.x;
	int stride_y = blockDim.y * gridDim.y;
	for (int i = index_x+1; i < N-1; i += stride_x)
	{
		for (int j = index_y+1; j < N-1; j += stride_y)
		{
/*
			// this one is 10% faster, but has some hidden bug...
			int output_index = N*(i-1) + (j-1);

			int input_index_0 = N*(i-1) + j-1;
			int input_index_1 = N*(i) + j-1;
			int input_index_2 = N*(i+1) + j-1;
			int input_index_3 = N*(i-1) + j;
			int input_index_5 = N*(i+1) + j;
			int input_index_6 = N*(i-1) + j+1;
			int input_index_7 = N*(i) + j+1;
			int input_index_8 = N*(i+1) + j+1;

			int mask_index_0 = (N-1)*(i-1) + j-1;
			int mask_index_1 = (N-1)*(i) + j-1;
			int mask_index_3 = (N-1)*(i-1) + j;
			int mask_index_4 = (N-1)*(i) + j;
			
			float mask_conv_diag = input_tensor[input_index_0] * mask_tensor[mask_index_0] 
							     + input_tensor[input_index_6] * mask_tensor[mask_index_3]
								 + input_tensor[input_index_2] * mask_tensor[mask_index_1] 
						  	 	 + input_tensor[input_index_8] * mask_tensor[mask_index_4];
			float mask_conv_side = input_tensor[input_index_3] * (mask_tensor[mask_index_0]+mask_tensor[mask_index_3])
  							 	 + input_tensor[input_index_1] * (mask_tensor[mask_index_0]+mask_tensor[mask_index_1])
  							 	 + input_tensor[input_index_7] * (mask_tensor[mask_index_3]+mask_tensor[mask_index_4])
  							 	 + input_tensor[input_index_5] * (mask_tensor[mask_index_1]+mask_tensor[mask_index_4]);
			float sum_resp_diag = input_tensor[input_index_0] + input_tensor[input_index_6] + input_tensor[input_index_2] + input_tensor[input_index_8];
			float sum_resp_side = input_tensor[input_index_3] + input_tensor[input_index_1] + input_tensor[input_index_7] + input_tensor[input_index_5];

			output_tensor[output_index] = mask_conv_diag *diag_coef_diff
										+ mask_conv_side *side_coef_diff/2
										+ sum_resp_diag * diag_coef_2
										+ sum_resp_side * diag_coef_2;
*/	
            output_tensor[(i-1) *N+ j-1]  = input_tensor[(i-1)*N+ j-1] * mask_tensor[(i-1) *N+ j-1] *diag_coef_1 
                                 + input_tensor[(i-1) *N+ j+1] * mask_tensor[(i-1) *N+ j] *diag_coef_1 
                                 + input_tensor[(i+1) *N+ j-1] * mask_tensor[(i) *N+ j-1] *diag_coef_1 
                                 + input_tensor[(i+1) *N+ j+1] * mask_tensor[(i) *N+ j] *diag_coef_1
                                 
                                 + input_tensor[(i-1) *N+ j] * (mask_tensor[(i-1) *N+ j-1] + mask_tensor[(i-1) *N+ j]) / 2. *side_coef_1 
                                 + input_tensor[(i) *N+ j-1] * (mask_tensor[(i-1) *N+ j-1] + mask_tensor[(i) *N+ j-1]) / 2. *side_coef_1
                                 + input_tensor[(i) *N+ j + 1] * (mask_tensor[(i-1) *N+ j] + mask_tensor[(i) *N+ j]) / 2. *side_coef_1
                                 + input_tensor[(i+1), j] * (mask_tensor[(i) *N+ j-1] + mask_tensor[(i) *N+ j]) / 2. *side_coef_1
             
                                 + input_tensor[(i-1) *N+ j-1] * (1-mask_tensor[(i-1) *N+ j-1]) *diag_coef_2 
                                 + input_tensor[(i-1) *N+ j+1] * (1-mask_tensor[(i-1) *N+ j])*diag_coef_2 
                                 + input_tensor[(i+1) *N+ j-1] * (1-mask_tensor[(i) *N+ j-1]) *diag_coef_2 
                                 + input_tensor[(i+1) *N+ j+1] * (1-mask_tensor[(i) *N+ j]) *diag_coef_2
            
                                 + input_tensor[(i-1) *N+ j] * (2-mask_tensor[(i-1) *N+ j-1] - mask_tensor[(i-1) *N+ j]) / 2. *side_coef_2 
                                 + input_tensor[(i) *N+ j-1] * (2-mask_tensor[(i-1) *N+ j-1] - mask_tensor[(i) *N+ j-1]) / 2. *side_coef_2
                                 + input_tensor[(i) *N+ j + 1] * (2-mask_tensor[(i-1) *N+ j] - mask_tensor[(i) *N+ j]) / 2. *side_coef_2
                                 + input_tensor[(i+1) *N+ j] * (2-mask_tensor[(i) *N+ j-1] - mask_tensor[(i) *N+ j]) / 2. *side_coef_2;
						
		}
	}
}
             
void mask_conv_gpu(int mat_row_num,  float *input_tensor, float *weights_tensor, float *output_tensor) {
	float diag_coef_1 = 16;
	float side_coef_1 = 16;
	float diag_coef_2 = 1;
	float side_coef_2 = 1;
	dim3 blocksize( 32, 32 );
	int bx = (mat_row_num+blocksize.x-1)/blocksize.x ;
	int by = (mat_row_num+blocksize.y-1)/blocksize.y ;
	dim3 gridsize(bx, by);
    _mask_conv<<<gridsize,blocksize>>>(diag_coef_1, diag_coef_2, side_coef_1, side_coef_2, mat_row_num, input_tensor, weights_tensor, output_tensor);
    cudaDeviceSynchronize();
}


void mask_conv_cpu(int N, float *input_tensor, float *weights_tensor, float * output_tensor)
{
	float diag_coef_1 = 16;
	float side_coef_1 = 16;
	float diag_coef_2 = 1;
	float side_coef_2 = 1;
	float diag_coef_diff = diag_coef_1 - diag_coef_2;
	float side_coef_diff = side_coef_1 - side_coef_2;

    for (int i = 1; i < N-1; i++) {
        for (int j = 1; j < N-1; j++) {

            output_tensor[(i-1) *N+ j-1]  = input_tensor[(i-1)*N+ j-1] * weights_tensor[(i-1) *N+ j-1] *diag_coef_1 
                                 + input_tensor[(i-1) *N+ j+1] * weights_tensor[(i-1) *N+ j] *diag_coef_1 
                                 + input_tensor[(i+1) *N+ j-1] * weights_tensor[(i) *N+ j-1] *diag_coef_1 
                                 + input_tensor[(i+1) *N+ j+1] * weights_tensor[(i) *N+ j] *diag_coef_1
                                 
                                 + input_tensor[(i-1) *N+ j] * (weights_tensor[(i-1) *N+ j-1] + weights_tensor[(i-1) *N+ j]) / 2. *side_coef_1 
                                 + input_tensor[(i) *N+ j-1] * (weights_tensor[(i-1) *N+ j-1] + weights_tensor[(i) *N+ j-1]) / 2. *side_coef_1
                                 + input_tensor[(i) *N+ j + 1] * (weights_tensor[(i-1) *N+ j] + weights_tensor[(i) *N+ j]) / 2. *side_coef_1
                                 + input_tensor[(i+1), j] * (weights_tensor[(i) *N+ j-1] + weights_tensor[(i) *N+ j]) / 2. *side_coef_1
             
                                 + input_tensor[(i-1) *N+ j-1] * (1-weights_tensor[(i-1) *N+ j-1]) *diag_coef_2 
                                 + input_tensor[(i-1) *N+ j+1] * (1-weights_tensor[(i-1) *N+ j])*diag_coef_2 
                                 + input_tensor[(i+1) *N+ j-1] * (1-weights_tensor[(i) *N+ j-1]) *diag_coef_2 
                                 + input_tensor[(i+1) *N+ j+1] * (1-weights_tensor[(i) *N+ j]) *diag_coef_2
            
                                 + input_tensor[(i-1) *N+ j] * (2-weights_tensor[(i-1) *N+ j-1] - weights_tensor[(i-1) *N+ j]) / 2. *side_coef_2 
                                 + input_tensor[(i) *N+ j-1] * (2-weights_tensor[(i-1) *N+ j-1] - weights_tensor[(i) *N+ j-1]) / 2. *side_coef_2
                                 + input_tensor[(i) *N+ j + 1] * (2-weights_tensor[(i-1) *N+ j] - weights_tensor[(i) *N+ j]) / 2. *side_coef_2
                                 + input_tensor[(i+1) *N+ j] * (2-weights_tensor[(i) *N+ j-1] - weights_tensor[(i) *N+ j]) / 2. *side_coef_2;
        }
   }
}


int main(void)
{
	int N = 2048;  
	float *resp, *mask, *load_gpu;
	float * load_cpu =  (float*) malloc(N*N * sizeof(float));

	// Allocate Unified Memory – accessible from CPU or GPU
	cudaMallocManaged(&resp, N*N*sizeof(float));
	cudaMallocManaged(&mask, (N-1)*(N-1)*sizeof(float));
	cudaMallocManaged(&load_gpu, N*N*sizeof(float));

	// assume they are row first stored
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			resp[i*N+j] = 1.0;
			if (i<N-1 & j<N-1){
				if (i<N/2)
					mask[i*N+j] = 1.0;
				else
					mask[i*N+j] = 0.0;
			}
		}
	}

	// Run kernel on CPU
    std::clock_t start = std::clock();
	mask_conv_cpu(N, resp, mask, load_cpu);
    double duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	std::cout<<"CPU time cost: "<<duration<<std::endl;

	// Run kernel on GPU
    start = std::clock();
	mask_conv_gpu(N, resp, mask, load_gpu);
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	std::cout<<"GPU time cost: "<<duration<<std::endl;

/*
	// check results
	ofstream myfile_cpu, myfile_gpu;
	myfile_cpu.open ("cpu.txt");
	myfile_gpu.open ("gpu.txt");
	for (int i = 0; i < N-2; i++){
		for (int j=0; j< N-2; j++){
			myfile_cpu <<load_cpu[N*i+j]<<" ";
			myfile_gpu <<load_gpu[N*i+j]<<" ";
		}
		myfile_cpu << std::endl;
		myfile_gpu << std::endl;
	}
	myfile_cpu.close();
	myfile_gpu.close();
*/
	// Free memory
	cudaFree(resp);
	cudaFree(mask);
	cudaFree(load_cpu);
	cudaFree(load_gpu);

	return 0;
}

//nvcc -arch=sm_61 -O3 mask_conv.cu -o mask_conv
/*
   			cpu 		gpu
256		0.001611	2.9e-05
1028 	0.037622	4.2e-05
*/

//nvcc mask_conv.cu -o mask_conv.cu.o 



