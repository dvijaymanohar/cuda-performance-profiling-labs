// Build: nvcc -O3 examples/launch_overhead.cu -o launch_overhead
#include <cuda_runtime.h>
#include <iostream>
__global__ void tiny(float*x){if(threadIdx.x==0&&blockIdx.x==0)*x+=1.0f;}
int main(){float*x;cudaMalloc(&x,4);cudaMemset(x,0,4);for(int n:{1,10,100,1000,10000}){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);for(int i=0;i<n;++i)tiny<<<1,1>>>(x);cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);std::cout<<"launches="<<n<<" total_ms="<<ms<<" us_per_launch="<<(ms*1000/n)<<"\n";cudaEventDestroy(a);cudaEventDestroy(b);}cudaFree(x);}
