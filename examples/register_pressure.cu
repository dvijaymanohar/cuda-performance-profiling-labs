// Build: nvcc -O3 -Xptxas=-v examples/register_pressure.cu -o register_pressure
#include <cuda_runtime.h>
#include <iostream>
template<int N>
__global__ void pressure(float*x,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 float v[N];#pragma unroll
 for(int k=0;k<N;++k)v[k]=x[i]+k;
 #pragma unroll
 for(int k=1;k<N;++k)v[0]=v[0]*1.00001f+v[k]*0.00001f;
 x[i]=v[0];
}
int main(){int n=1<<22;float*x;cudaMalloc(&x,(size_t)n*4);cudaMemset(x,0,(size_t)n*4);pressure<8><<<(n+255)/256,256>>>(x,n);pressure<64><<<(n+255)/256,256>>>(x,n);cudaDeviceSynchronize();std::cout<<"Compile with -Xptxas=-v and profile pressure<8> vs pressure<64> for registers, spills, occupancy, stalls.\n";cudaFree(x);}
