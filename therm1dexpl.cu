#include <iostream>
#include <fstream>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
//#define WRITE_TO_FILE
using namespace std;

//Обработчик ошибок
static void HandleError(cudaError_t err,
                        const char *file,
                        int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
            file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR( error ) (HandleError( error, __FILE__, __LINE__ ))

//Ядро программы
__global__ void stepKernel(float *Tdev,float *Tplusdev,float h,float tau,int N)
{
 int tid=blockIdx.x*blockDim.x+threadIdx.x;
 if(tid==0)
 {
	 Tplusdev[tid]=0.0;
 }
 else if(tid==N-1)
 {
	 Tplusdev[tid]=tau/h*((-Tdev[tid]+Tdev[tid-1])/h+5.0)+Tdev[tid];
 }
 else if(tid<N-1)
 {
 Tplusdev[tid]=tau/h/h*(Tdev[tid+1]-2.0*Tdev[tid]+Tdev[tid-1])+Tdev[tid];
 }
}

int main()
{
#ifdef WRITE_TO_FILE
   ofstream ofile("../therm1dexpl/data.dat");
   ofile.precision(16);
   int counter=0, writeeach=1;
#endif

   int N=101;
   float L=10.0,tau=0.001,tmax=5.0,t=0.0;
   float h=L/N;
   float *T, *Tplus,*Tdev,*Tplusdev,*temp;
   float cputime,gputime;

   T=new float[N];
   Tplus=new float[N];

   HANDLE_ERROR( cudaMalloc(&Tdev,N*sizeof(float)) );
   HANDLE_ERROR( cudaMalloc(&Tplusdev,N*sizeof(float)) );

   HANDLE_ERROR( cudaMemset(Tdev,0,N*sizeof(float)) );
   memset(T,0,N*sizeof(float));

   dim3 threads(1024,1,1);
   dim3 blocks((N%1024==0)?(N/1024):(N/1024+1),1,1);

   cudaEvent_t start,end;
   HANDLE_ERROR( cudaEventCreate(&start) );
   HANDLE_ERROR( cudaEventCreate(&end) );

   HANDLE_ERROR( cudaEventRecord(start) );
   HANDLE_ERROR( cudaEventSynchronize(start) );
   while(t<tmax-tau/2.0)
   {
   stepKernel<<<blocks,threads>>>(Tdev,Tplusdev,h,tau,N);
   HANDLE_ERROR( cudaGetLastError() );
   HANDLE_ERROR( cudaDeviceSynchronize() );

   temp=Tdev;
   Tdev=Tplusdev;
   Tplusdev=temp;
   t+=tau;
#ifdef WRITE_TO_FILE
   HANDLE_ERROR( cudaMemcpy(T,Tdev,N*sizeof(float),cudaMemcpyDeviceToHost) );
   if(counter%writeeach==0)
   {
       for(int i=0;i<N;i++)
           ofile<<T[i]<<endl;
       ofile<<endl;
       ofile<<endl;
   }
   counter++;
#endif
   }
   HANDLE_ERROR( cudaMemcpy(T,Tdev,N*sizeof(float),cudaMemcpyDeviceToHost) );
   HANDLE_ERROR( cudaEventRecord(end) );
   HANDLE_ERROR( cudaEventSynchronize(end) );
   HANDLE_ERROR( cudaEventElapsedTime(&gputime,start,end) );
   gputime/=1000.0;

int cl=0;
cl-=clock();
t=0;
   while(t<tmax-tau/2.0)
      {

	   	 Tplus[0]=0.0;
	   	 Tplus[N-1]=tau/h*((-T[N-1]+T[N-2])/h+5.0)+T[N-1];
	   	 for(int i=1;i<N-1;i++)
	     Tplus[i]=tau/h/h*(T[i+1]-2.0*T[i]+T[i-1])+T[i];

      t+=tau;
      temp=T;
         T=Tplus;
         Tplus=temp;

      }
   cl+=clock();
   cputime=(float)cl/CLOCKS_PER_SEC;

   cout<<"CPU time: "<<cputime<<endl;
   cout<<"GPU time: "<<gputime<<endl;
   cout<<"Ratio: "<<cputime/gputime<<endl;
#ifdef WRITE_TO_FILE
   ofile.close();
#endif
   HANDLE_ERROR( cudaFree(Tdev) );
   HANDLE_ERROR( cudaFree(Tplusdev) );
   HANDLE_ERROR( cudaEventDestroy(start) );
   HANDLE_ERROR( cudaEventDestroy(end) );
   delete[] T;
   delete[] Tplus;
   return 0;
}
