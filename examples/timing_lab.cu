#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

__global__ void saxpy(float* y,const float* x,float a,size_t n){
    size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)y[i]=a*x[i]+y[i];
}
int main(int argc,char** argv){
    size_t n=argc>1?std::strtoull(argv[1],nullptr,10):1<<24;
    float *x,*y; cudaMalloc(&x,n*4);cudaMalloc(&y,n*4);cudaMemset(x,0,n*4);cudaMemset(y,0,n*4);
    int blocks=(n+255)/256;
    for(int i=0;i<5;++i)saxpy<<<blocks,256>>>(y,x,2.0f,n);
    cudaDeviceSynchronize();
    std::vector<float> samples;
    for(int r=0;r<11;++r){
        cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);
        cudaEventRecord(a);saxpy<<<blocks,256>>>(y,x,2.0f,n);cudaEventRecord(b);cudaEventSynchronize(b);
        float ms;cudaEventElapsedTime(&ms,a,b);samples.push_back(ms);cudaEventDestroy(a);cudaEventDestroy(b);
    }
    std::sort(samples.begin(),samples.end());
    std::cout<<"median_ms="<<samples[samples.size()/2]<<" min_ms="<<samples.front()<<" max_ms="<<samples.back()<<"\n";
    cudaFree(x);cudaFree(y);
}
