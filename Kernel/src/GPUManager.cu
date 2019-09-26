/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "GPUManager.h"
#include <cstdio>
#include "GooStatsException.h"
bool GPUManager::preinit() {
  if(!report()) {
    throw GooStatsException("Cannot find a free GPU!");
  }
  return true;
}
#ifdef __CUDACC__
bool GPUManager::report(bool siliently) const {
  int devicesCount(1);
  cudaGetDeviceCount(&devicesCount);
  if(devicesCount>1000) {
    printf("number of devices larger than 1000. this is suspicious. Please modify the code here or login to the GPU node.");
    return false;
  }
  for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
    printf("Checking GPU [%d]/[%d]\n",deviceIndex,devicesCount);
    if(report(deviceIndex,siliently)) return true;
  }
  return false;
#else
bool GPUManager::report(bool ) const {
  return true;
#endif
}
#ifdef __CUDACC__
bool GPUManager::report(int gpu_id,bool siliently) const {
  cudaSetDevice(gpu_id);
  cudaError_t cuda_status;
  //// show memory usage of GPU
  //size_t free_byte ;
  //size_t total_byte ;
  //cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
  //if ( cudaSuccess != cuda_status ){
  //  printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
  //  return false;
  //}
  //if(!siliently) {
  //  double free_db = (double)free_byte ;
  //  double total_db = (double)total_byte ;
  //  double used_db = total_db - free_db ;
  //  printf("GPU memory usage: used = %lf, free = %lf MB, total = %lf MB\n",
  //      used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
  //}
  int device;
  cuda_status = cudaGetDevice( &device );
  if( cudaSuccess != cuda_status ) {
    printf("Error: cudaGetDevice fails, %s \n", cudaGetErrorString(cuda_status) );
    return false;
  }
  cudaDeviceProp prop;
  cuda_status = cudaGetDeviceProperties(&prop, device);
  if( cudaSuccess != cuda_status ) {
    printf("Error: cudaGetDeviceProperties fails, %s \n", cudaGetErrorString(cuda_status) );
    return false;
  }
	int *test;
	cuda_status = cudaMalloc( (void**)&test, sizeof(int)); 
  if( cudaSuccess != cuda_status ) {
    printf("Error: cudaMalloc fails, %s \n", cudaGetErrorString(cuda_status) );
    return false;
  }
	cudaFree( test );
  if(!siliently) {
    printf("Running on [%s] with compute capability of %d.%d\n",prop.name,prop.major,prop.minor);
  }
//  if( !(prop.major>3 || prop.major==3 && prop.minor>=5) )
//    return false;
#else
bool GPUManager::report(int ,bool ) const {
#endif
  return true;
}
