/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "GooStatsSumPdf.h"

//EXEC_TARGET fptype device_SumPdfsExt_withSys (fptype* evt, fptype* p, unsigned int* indices) { 
//  const int cIndex = RO_CACHE(indices[1]); 
//  const fptype npe_val = evt[RO_CACHE(indices[2 + RO_CACHE(indices[0])])]; 
//  const fptype npe_lo = RO_CACHE(functorConstants[cIndex]);
//  const fptype npe_step = RO_CACHE(functorConstants[cIndex+1]);
//  const int npe_bin = (int) FLOOR((npe_val-npe_lo)/npe_step); // no problem with FLOOR: start from 0.5, which corresponse to bin=0
//  const int Ncomps = RO_CACHE(indices[2]); 
//  fptype ret = 0;
//  for (int par = 0; par < Ncomps; ++par) {
//    const int workSpaceIndex = RO_CACHE(indices[par+3]);
//    const fptype weight = RO_CACHE(p[RO_CACHE(indices[par+3+Ncomps])]);
//    const int i = RO_CACHE(indices[par+3+Ncomps+Ncomps]);
//    const fptype sysi = i?RO_CACHE(p[i]):1;
//    const fptype curr = dev_componentWorkSpace[workSpaceIndex][npe_bin];
//    ret += weight * sysi * curr; // normalization is always 1
//#ifdef convolution_CHECK
//    if(npe_bin==0)
//      //  if(THREADIDX==10)
//      printf("+ npe %.1lf npebin %d ret %.10lf w %.10lf wk %d cur %.10lf sysi %d sys %.10lf\n",npe_val,npe_bin,ret,weight,workSpaceIndex,curr,i,sysi);
//#endif
//  }
//  const int i = RO_CACHE(indices[Ncomps+3+Ncomps+Ncomps]);
//  const fptype sys = i?RO_CACHE(p[i]):1;
//  ret *= RO_CACHE(functorConstants[cIndex+2]) * sys; // exposure
//#ifdef convolution_CHECK
//    if(npe_bin==0)
//      //  if(THREADIDX==10)
//      printf("+ npe %.1lf npebin %d ret %.10lf sysi %d sys %.10lf\n",npe_val,npe_bin,ret,i,sys);
//#endif
//  //  printf("npe %.1lf ret %.10lf\n",npe_val,ret);
//
//  return ret; 
//}
//
//EXEC_TARGET fptype device_SumPdfsExt (fptype* evt, fptype* p, unsigned int* indices) { 
//  const int cIndex = RO_CACHE(indices[1]); 
//  const fptype npe_val = evt[RO_CACHE(indices[2 + RO_CACHE(indices[0])])]; 
//  const fptype npe_lo = RO_CACHE(functorConstants[cIndex]);
//  const fptype npe_step = RO_CACHE(functorConstants[cIndex+1]);
//  const int npe_bin = (int) FLOOR((npe_val-npe_lo)/npe_step); // no problem with FLOOR: start from 0.5, which corresponse to bin=0
//  const int Ncomps = RO_CACHE(indices[2]); 
//  fptype ret = 0;
//  for (int par = 0; par < Ncomps; ++par) {
//    const int workSpaceIndex = RO_CACHE(indices[par+3]);
//    const fptype weight = RO_CACHE(p[RO_CACHE(indices[par+3+Ncomps])]);
//    const fptype curr = dev_componentWorkSpace[workSpaceIndex][npe_bin];
//    ret += weight * curr; // normalization is always 1
//#ifdef convolution_CHECK
//    if(npe_bin==0)
//      //  if(THREADIDX==10)
//      printf("+ npe %.1lf npebin %d ret %.10lf w %.10lf wk %d cur %.10lf\n",npe_val,npe_bin,ret,weight,workSpaceIndex,curr);
//#endif
//  }
//  ret *= RO_CACHE(functorConstants[cIndex+2]); // exposure
//  //  printf("npe %.1lf ret %.10lf\n",npe_val,ret);
//
//  return ret; 
//}
//EXEC_TARGET fptype device_SumPdfsExtMask (fptype* evt, fptype* , unsigned int* indices) { 
//  const int cIndex = RO_CACHE(indices[1]); 
//  const fptype npe_val = evt[RO_CACHE(indices[2 + RO_CACHE(indices[0])])]; 
//  const fptype npe_lo = RO_CACHE(functorConstants[cIndex]);
//  const fptype npe_step = RO_CACHE(functorConstants[cIndex+1]);
//  const int npe_bin = (int) FLOOR((npe_val-npe_lo)/npe_step); // no problem with FLOOR: start from 0.5, which corresponse to bin=0
//  const int Ncomps = RO_CACHE(indices[2]); 
//  fptype ret = 0;
//#ifdef convolution_CHECK
//  if(npe_bin==0)
//    printf("cIndex %d npe %.5le npelo %.5le npestep %.5le npebin %d Ncomps %d \n",cIndex,npe_val,npe_lo,npe_step,npe_bin,Ncomps);
//#endif
//  for (int par = 0; par < Ncomps; ++par) {
//    const int workSpaceIndex = RO_CACHE(indices[par+3]);
//    const int maskSpaceIndex = RO_CACHE(indices[par+3+Ncomps]);
//    const fptype curr = dev_componentWorkSpace[workSpaceIndex][npe_bin];
//    const fptype mask = RO_CACHE(dev_raw_masks[maskSpaceIndex][npe_bin]);
//    ret += curr*mask; // normalization is always 1
//#ifdef Mask_CHECK
//    //  if(THREADIDX==0)
//    printf("npe %.1lf npebin %d wid %d mid %d w %.14le m %.14le curr %.14le ret %.14le\n",npe_val,npe_bin,workSpaceIndex,maskSpaceIndex,curr,mask,curr*mask,ret);
//#endif
//  }
//  // ret *= RO_CACHE(functorConstants[cIndex+2]); // exposure
//
//  return ret; 
//}
//EXEC_TARGET fptype device_SumPdfsExtSimple (fptype* evt, fptype* , unsigned int* indices) { 
//  const int cIndex = RO_CACHE(indices[1]); 
//  const fptype npe_val = evt[RO_CACHE(indices[2 + RO_CACHE(indices[0])])]; 
//  const fptype npe_lo = RO_CACHE(functorConstants[cIndex]);
//  const fptype npe_step = RO_CACHE(functorConstants[cIndex+1]);
//  const int npe_bin = (int) FLOOR((npe_val-npe_lo)/npe_step); // no problem with FLOOR: start from 0.5, which corresponse to bin=0
//  const int Ncomps = RO_CACHE(indices[2]); 
//  fptype ret = 0;
//  for (int par = 0; par < Ncomps; ++par) {
//    const int workSpaceIndex = RO_CACHE(indices[par+3]);
//    const fptype curr = dev_componentWorkSpace[workSpaceIndex][npe_bin];
//    ret += curr; // normalization is always 1
//  }
//  //  ret *= RO_CACHE(functorConstants[cIndex+2]); // exposure
//
//  return ret; 
//}


#ifdef NLL_CHECK
#include "GooStatsNLLCheck.h"
__host__ double SumPdf::sumOfNll (int numVars) const {
  static thrust::plus<fptype> cudaPlus;
  thrust::constant_iterator<int> eventSize(numVars); 
  thrust::constant_iterator<fptype*> arrayAddress(dev_event_array[pdfId]); 
  fptype dummy = 0;
  thrust::counting_iterator<int> eventIndex(0); 
  DEVICE_VECTOR<fptype> results(numEntries);
  assert(numEntries);
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(eventIndex, arrayAddress, eventSize)),
      thrust::make_zip_iterator(thrust::make_tuple(eventIndex + numEntries, arrayAddress, eventSize)),
      results.begin(),
      *logger);
  // debug start
  static double sum = 0;
  static bool first = false;
  thrust::host_vector<fptype> logL = results; 
  DEVICE_VECTOR<fptype> dev_sumV(numEntries);
  MetricTaker modalor((PdfBase*)(this), getMetricPointer("ptr_to_Eval")); 
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(eventIndex, arrayAddress, eventSize)),
      thrust::make_zip_iterator(thrust::make_tuple(eventIndex + numEntries, arrayAddress, eventSize)),
      dev_sumV.begin(),
      modalor);
  thrust::host_vector<fptype> sumV = dev_sumV;
  thrust::host_vector<fptype> fVal[200];
  for(unsigned int j = 0;j<components.size();++j) {
    const int workSpaceIndex = funMap.at(components.at(j));
    fVal[workSpaceIndex] = *(componentWorkSpace[workSpaceIndex]);
  }
  fptype* host_array = new fptype[numEntries*3];
  MEMCPY(host_array,dev_event_array[pdfId], 3*numEntries*sizeof(fptype), cudaMemcpyDeviceToHost);
  for(unsigned int i = 0;i<logL.size();++i) {
    double binVolume = host_array[i*3+2];
    sum+= logL[i];
//    printf("log(L) %.12le b %lf M %lf tot %.12le\n",sum,host_array[i*3],host_array[i*3+1],sumV[i]*binVolume);
    GooStatsNLLCheck::get()->record_LL(i,host_array[i*3],host_array[i*3+1],sumV[i]*binVolume,logL[i]);
    for(unsigned int j = 0;j<components.size();++j) {
      const int workSpaceIndex = funMap.at(components.at(j));
      double result = fVal[workSpaceIndex][i]*norm*binVolume*host_params[host_indices[parameters+3+j+components.size()]];
//      printf(" %s %.12le",components.at(j)->getName().c_str(),result);
      GooStatsNLLCheck::get()->record_species(i,components.at(j)->getName(),result);
    }
//    printf("\n");
  }
  if(first) { sum = 0; first = false; } else first = true;
  delete [] host_array;
  // debug end
  return thrust::reduce(results.begin(),results.end(),dummy,cudaPlus);
}
#endif
