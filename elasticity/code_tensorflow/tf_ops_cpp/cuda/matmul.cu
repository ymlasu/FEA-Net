#include <stdio.h>
#include <cuda.h>
#include <time.h>

__global__
void kernel(float *vec, float *mat, float *out, const int N, const int M){
    int tid=threadIdx.x+blockIdx.x*blockDim.x;
        float sum=0;
    if(tid<M){
        for(int i=0; i<N; i++)
            sum += vec[i]*mat[(i*M)+tid];
        out[tid]=sum;
    }
}

// debuging functions
void init_array(float *a, const int N);
void init_mat(float *a, const int N, const int M);
void print_array(float *a, const int N, char *d);
void print_mat(float *a, const int N, const int M, char *d);

int main (void) {
        srand( time(NULL) );

    float *a, *b, *c;
        float *dev_a, *dev_b, *dev_c;

    int N=3;
    int M=4;
    a=(float*)malloc(sizeof(float)*N);
    b=(float*)malloc(sizeof(float)*N*M);
    c=(float*)malloc(sizeof(float)*M);
        init_array(a, N);
        init_mat(b, N, M);
        init_array(c, M);

    printf("<<<<<<<<<< initial data:\n");
        print_array(a, N, "in-vector");
        print_mat(b, N, M, "matrix");
        print_array(c, M, "out-vector");

        cudaMalloc((void**)&dev_a, sizeof(float)*N);
        cudaMalloc((void**)&dev_b, sizeof(float)*N*M);
        cudaMalloc((void**)&dev_c, sizeof(float)*M);

        cudaMemcpy(dev_a, a, sizeof(float)*N, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, sizeof(float)*N*M, cudaMemcpyHostToDevice);

    printf("\n\nRunning Kernel...\n\n");
        kernel<<<M/256+1, 256>>>(dev_a, dev_b, dev_c, N, M);
        //printf("error code: %s\n",cudaGetErrorString(cudaGetLastError()));

        cudaMemcpy(c, dev_c, sizeof(float)*M, cudaMemcpyDeviceToHost);

        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);

    printf(">>>>>>>>>> final data:\n");
        print_array(c, M, "out-vector");

        return 0;
};

void init_array(float *a, const int N) {
        int i;
        for(i=0; i<N; i++)
                a[i] = rand() % 4 + 1;
}
void init_mat(float *a, const int N, const int M) {
        int i, j;
        for(i=0; i<N; i++)
            for(j=0; j<M; j++)
                    a[i*M+j] = rand() % 4 + 1;
}
void print_array(float *a, const int N, char *d) {
        int i;
        for(i=0; i<N; i++)
                printf("\n%s[%d]: %f",d, i, a[i]);
    printf("\n");
}
void print_mat(float *a, const int N, const int M, char *d) {
        int i, j;
        for(i=0; i<N; i++){
        printf("\n%s[%d]:", d, i);
        for (j=0; j<M; j++)
                    printf("\t%6.4f", a[i*M+j]);
    }
    printf("\n");
}
