// Build: nvcc -O3 examples/bandwidth_copy.cu -o bandwidth_copy
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
__global__ void copy_kernel(const float*a,float*b,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x)b[i]=a[i];}
int main(int argc,char**argv){size_t n=argc>1?std::strtoull(argv[1],nullptr,10):(1ull<<26);float*a,*b;cudaMalloc(&a,n*4);cudaMalloc(&b,n*4);cudaMemset(a,1,n*4);for(int w=0;w<3;++w)copy_kernel<<<4096,256>>>(a,b,n);cudaDeviceSynchronize();cudaEvent_t x,y;cudaEventCreate(&x);cudaEventCreate(&y);cudaEventRecord(x);copy_kernel<<<4096,256>>>(a,b,n);cudaEventRecord(y);cudaEventSynchronize(y);float ms;cudaEventElapsedTime(&ms,x,y);double gb=2.0*n*sizeof(float)/1e9;std::cout<<"ms="<<ms<<" effective_GBps="<<(gb/(ms/1000))<<"\n";cudaFree(a);cudaFree(b);}
