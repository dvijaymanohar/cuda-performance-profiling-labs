// Build: nvcc -O3 examples/gemm_naive_tiled.cu -o gemm_naive_tiled
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>
__global__ void naive(const float*A,const float*B,float*C,int n){int r=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;if(r<n&&c<n){float s=0;for(int k=0;k<n;++k)s+=A[r*n+k]*B[k*n+c];C[r*n+c]=s;}}
__global__ void tiled(const float*A,const float*B,float*C,int n){__shared__ float as[16][16],bs[16][16];int r=blockIdx.y*16+threadIdx.y,c=blockIdx.x*16+threadIdx.x;float s=0;for(int t=0;t<(n+15)/16;++t){int ak=t*16+threadIdx.x,bk=t*16+threadIdx.y;as[threadIdx.y][threadIdx.x]=(r<n&&ak<n)?A[r*n+ak]:0;bs[threadIdx.y][threadIdx.x]=(bk<n&&c<n)?B[bk*n+c]:0;__syncthreads();for(int k=0;k<16;++k)s+=as[threadIdx.y][k]*bs[k][threadIdx.x];__syncthreads();}if(r<n&&c<n)C[r*n+c]=s;}
int main(){int n=512;size_t bytes=(size_t)n*n*4;std::vector<float>h(n*n,1),a(n*n),b(n*n);float *A,*B,*C;cudaMalloc(&A,bytes);cudaMalloc(&B,bytes);cudaMalloc(&C,bytes);cudaMemcpy(A,h.data(),bytes,cudaMemcpyHostToDevice);cudaMemcpy(B,h.data(),bytes,cudaMemcpyHostToDevice);dim3 th(16,16),bl((n+15)/16,(n+15)/16);naive<<<bl,th>>>(A,B,C,n);cudaMemcpy(a.data(),C,bytes,cudaMemcpyDeviceToHost);tiled<<<bl,th>>>(A,B,C,n);cudaMemcpy(b.data(),C,bytes,cudaMemcpyDeviceToHost);bool ok=true;for(size_t i=0;i<a.size();++i)if(std::fabs(a[i]-b[i])>1e-2){ok=false;break;}std::cout<<(ok?"PASS":"FAIL")<<". Use Nsight Compute to compare memory traffic and arithmetic intensity.\n";cudaFree(A);cudaFree(B);cudaFree(C);return ok?0:2;}
