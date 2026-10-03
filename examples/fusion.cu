// Build: nvcc -O3 examples/fusion.cu -o fusion
#include <cuda_runtime.h>
#include <iostream>
__global__ void bias(float*x,float b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]+=b;}
__global__ void relu(float*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=x[i]>0?x[i]:0;}
__global__ void fused(float*x,float b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=x[i]+b;x[i]=v>0?v:0;}}
int main(){int n=1<<24;float*x;cudaMalloc(&x,(size_t)n*4);cudaMemset(x,0,(size_t)n*4);int bl=(n+255)/256;bias<<<bl,256>>>(x,-.5f,n);relu<<<bl,256>>>(x,n);fused<<<bl,256>>>(x,-.5f,n);cudaDeviceSynchronize();std::cout<<"Profile 2-kernel bias+relu vs fused kernel. Compare launches, global traffic, registers, and end-to-end impact.\n";cudaFree(x);}
