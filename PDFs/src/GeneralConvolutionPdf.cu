/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "GeneralConvolutionPdf.h"
int totalGeneralConvolutions = 0; 

#define GEN_CONVOLUTION_CACHE_SIZE 768u
// I would really like this to be 1024, which is the maximum number of threads
// in a block (for compute capability 2.0 and up). Unfortunately this causes 
// the program to hang, presumably because there isn't enough memory and something
// goes wrong. So... 512 should be enough for anyone, right? 

// Need multiple working spaces for the case of several convolutions in one PDF. 
MEM_CONSTANT fptype* dev_modWorkSpace_general[100];
MEM_CONSTANT fptype* dev_resWorkSpace_general[100]; 

EXEC_TARGET fptype device_ConvolvePdfs_general (fptype* evt, fptype* , unsigned int* indices) { 
  const fptype obs_val = evt[RO_CACHE(indices[2 + RO_CACHE(indices[0])])]; // ok

  const int cIndex = RO_CACHE(indices[5]);
  const int workSpaceIndex = RO_CACHE(indices[6]); // ok
  const int obs_numbins = RO_CACHE(indices[7]); // ok
  const int intvar_numbins = RO_CACHE(indices[8]); // ok

  const fptype obs_lo = RO_CACHE(functorConstants[cIndex]); // ok
  const fptype obs_step = RO_CACHE(functorConstants[cIndex+1]); // ok
  const fptype intvar_step = RO_CACHE(functorConstants[cIndex+2]); // ok

  const int obs_bin = (int) FLOOR((obs_val-obs_lo)/obs_step);
  unsigned int arr_begin = 0;
  while(RO_CACHE(dev_resWorkSpace_general[workSpaceIndex][obs_bin+arr_begin*obs_numbins])<-1) ++arr_begin; // ugly. use -1 as the secret message..
  arr_begin += (intvar_numbins-arr_begin+1)%2;

  const fptype model_first = RO_CACHE(dev_modWorkSpace_general[workSpaceIndex][arr_begin]);
  fptype ret     = model_first*(model_first==0?0:RO_CACHE(dev_resWorkSpace_general[workSpaceIndex][obs_bin+arr_begin*obs_numbins]));
//#ifdef convolution_CHECK
//  { int intvar_bin = arr_begin; const fptype model = model_first; const fptype resol = model_first==0?0:RO_CACHE(dev_resWorkSpace_general[workSpaceIndex][obs_bin+arr_begin*obs_numbins]);
//    int factor = 1;
//    if(obs_val==300.5)
//      printf("%s npe %.1lf e %d M %.10le R %.10le F %.10le sum %.10le\n",__func__,obs_val,intvar_bin,model,resol,factor*model*resol/3*intvar_step,ret/3*intvar_step); }
//#endif
  // example
  // intvar_numbins = 100, first non-zero i: 3
  // 3,4,... 99 -> 97 bins ok
  // weight
  // 1 2 4 2 4 2 ... 1

  const int intvar_lastval = intvar_numbins-2;
  for (int intvar_bin = arr_begin+1; intvar_bin <= intvar_lastval ; ++intvar_bin) {
    const int factor = (((intvar_bin - arr_begin) % 2) ? 4 : 2);
    const fptype model = RO_CACHE(dev_modWorkSpace_general[workSpaceIndex][intvar_bin]); 
    const fptype resol = (model == 0) ? 0: RO_CACHE(dev_resWorkSpace_general[workSpaceIndex][obs_bin+intvar_bin*obs_numbins]); 
    ret += factor * model*resol;
//#ifdef convolution_CHECK
////  if(THREADIDX==10)
//    if(obs_val==300.5)
//      printf("%s npe %.1lf e %d M %.10le R %.10le F %.10le sum %.10le\n",__func__,obs_val,intvar_bin,model,resol,factor*model*resol/3*intvar_step,ret/3*intvar_step);
//#endif
  }
  const fptype model_last = RO_CACHE(dev_modWorkSpace_general[workSpaceIndex][intvar_numbins-1]);
  ret    += model_last*(model_last==0 ? 0 : RO_CACHE(dev_resWorkSpace_general[workSpaceIndex][obs_bin+(intvar_numbins-1)*obs_numbins]));
#ifdef convolution_CHECK
  { int intvar_bin = (intvar_numbins-1); const fptype model = model_last; const fptype resol = model_last==0?0:RO_CACHE(dev_resWorkSpace_general[workSpaceIndex][obs_bin+(intvar_numbins-1)*obs_numbins]);
    int factor = 1;
    if(obs_val==300.5)
      printf("%s npe %.1lf e %d M %.10le R %.10le F %.10le sum %.10le\n",__func__,obs_val,intvar_bin,model,resol,factor*model*resol/3*intvar_step,ret/3*intvar_step); }
#endif
  ret /= 3;
  ret *= intvar_step;
  return ret; 
}

EXEC_TARGET fptype device_ConvolveSharedPdfs_general (fptype* evt, fptype* , unsigned int* indices) { 
  const fptype obs_val = evt[RO_CACHE(indices[2 + RO_CACHE(indices[0])])]; // ok

  const int cIndex = RO_CACHE(indices[5]); // ok
  const int workSpaceIndex = RO_CACHE(indices[6]); // ok
  const int obs_numbins = RO_CACHE(indices[7]); // ok
  const int intvar_numbins = RO_CACHE(indices[8]); // ok

  const fptype obs_lo = RO_CACHE(functorConstants[cIndex]); // ok
  const fptype obs_step = RO_CACHE(functorConstants[cIndex+1]); // ok
  const fptype intvar_step = RO_CACHE(functorConstants[cIndex+2]); // ok

  const int obs_bin = (int) FLOOR((obs_val-obs_lo)/obs_step); // ok
  unsigned int arr_begin = 0;
  while(RO_CACHE(dev_resWorkSpace_general[workSpaceIndex][obs_bin+arr_begin*obs_numbins])<-1) ++arr_begin; // ugly. use -1 as the secret message..
  arr_begin += (intvar_numbins-arr_begin+1)%2;

  const fptype model_first = RO_CACHE(dev_modWorkSpace_general[workSpaceIndex][arr_begin]);
  fptype ret     = model_first*(model_first==0?0:RO_CACHE(dev_resWorkSpace_general[workSpaceIndex][obs_bin+arr_begin*obs_numbins]));
#ifdef convolution_CHECK
  { int intvar_bin = arr_begin; const fptype model = model_first; const fptype resol = model_first==0?0:RO_CACHE(dev_resWorkSpace_general[workSpaceIndex][obs_bin+arr_begin*obs_numbins]);
    int factor = 1;
    if(obs_val==300.5)
      printf("%s npe %.1lf e %d M %.10le R %.10le F %.10le sum %.10le\n",__func__,obs_val,intvar_bin,model,resol,factor*model*resol/3*intvar_step,ret/3*intvar_step); }
#endif

  MEM_SHARED static fptype modelCache[GEN_CONVOLUTION_CACHE_SIZE]; 
  // Or number of loaders, or available threads.
  // for the last block, we have only (obs_numbins%BLOCKDIM) threads
  const int numToLoad = min(GEN_CONVOLUTION_CACHE_SIZE , (BLOCKIDX<GRIDDIM-1)?static_cast<unsigned int>(BLOCKDIM):static_cast<unsigned int>(obs_numbins%BLOCKDIM));
  // We have intvar_numbins to load while we have numToLoad threads.
  const int intvar_lastval = intvar_numbins-2;
  for (int intvar_bin_blockstart = arr_begin; intvar_bin_blockstart <= intvar_lastval; intvar_bin_blockstart+= numToLoad) {
    if (THREADIDX < numToLoad) { 
      const int intvar_bin = intvar_bin_blockstart+THREADIDX;
      if(intvar_bin<=intvar_lastval) {
        modelCache[THREADIDX] = RO_CACHE(dev_modWorkSpace_general[workSpaceIndex][intvar_bin]);
      }
    }
    THREAD_SYNCH 
    // now model[intvar_bin_blockstart] ~ model[intvar_bin_blockstart+numToLoad-1] is loaded.
    for (int intvar_bin_inBlock = 0; intvar_bin_inBlock < numToLoad; ++intvar_bin_inBlock) {
      const int intvar_bin = intvar_bin_blockstart + intvar_bin_inBlock;
      if (intvar_bin > intvar_lastval) break; 
      const fptype model = modelCache[intvar_bin_inBlock]; 
      const fptype resol = (model == 0) ? 0: RO_CACHE(dev_resWorkSpace_general[workSpaceIndex][obs_bin+intvar_bin*obs_numbins]); 
      const unsigned int factor = (((intvar_bin - arr_begin) % 2) ? 4 : 2);
      ret += factor * model*resol;
    }
    THREAD_SYNCH // if you do not synch here, some thread will go to the next loop
  }
  const fptype model_last = RO_CACHE(dev_modWorkSpace_general[workSpaceIndex][intvar_numbins-1]);
  ret    += model_last*(model_last==0 ? 0 : RO_CACHE(dev_resWorkSpace_general[workSpaceIndex][obs_bin+(intvar_numbins-1)*obs_numbins]));
//#ifdef convolution_CHECK
//  { int intvar_bin = (intvar_numbins-1); const fptype model = model_last; const fptype resol = model_last==0?0:RO_CACHE(dev_resWorkSpace_general[workSpaceIndex][obs_bin+(intvar_numbins-1)*obs_numbins]);
//    int factor = 1;
//    if(obs_val==300.5)
//      printf("%s npe %.1lf e %d M %.10le R %.10le F %.10le sum %.10le\n",__func__,obs_val,intvar_bin,model,resol,factor*model*resol/3*intvar_step,ret/3*intvar_step); }
//#endif
  ret /= 3;
  ret *= intvar_step;
  return ret; 
}

MEM_DEVICE device_function_ptr ptr_to_ConvolvePdfs_general = device_ConvolvePdfs_general; 
MEM_DEVICE device_function_ptr ptr_to_ConvolveSharedPdfs_general = device_ConvolveSharedPdfs_general; 

GeneralConvolutionPdf::GeneralConvolutionPdf (std::string n,
    Variable* obs, 
    Variable* intvar, 
    GooPdf* m, 
    GooPdf* r, bool syn_loading) 
  : GooPdf(obs, n)
  , model(m)
  , resolution(r)
  , host_iConsts(0LL)
  , modelWorkSpace(0LL)
  , resolWorkSpace(0LL)
    , workSpaceIndex(0)
{
  assert(obs);
  assert(intvar);
  // Constructor for convolution without cooperative
  // loading of model cache. This is slow, but conceptually
  // simple. 
  components.push_back(model);
  components.push_back(resolution);

  // Indices stores (function index)(parameter index) doublet for model and resolution function. 
  std::vector<unsigned int> paramIndices;
  paramIndices.push_back(model->getFunctionIndex()); // 1
  paramIndices.push_back(model->getParameterIndex()); // 2
  paramIndices.push_back(resolution->getFunctionIndex()); // 3
  paramIndices.push_back(resolution->getParameterIndex()); // 4
  paramIndices.push_back(registerConstants(3)); // 5
  paramIndices.push_back(workSpaceIndex = totalGeneralConvolutions++); // 6
  paramIndices.push_back(obs->numbins); // 7
  paramIndices.push_back(intvar->numbins); // 8

  if(syn_loading)
    GET_FUNCTION_ADDR(ptr_to_ConvolveSharedPdfs_general);
  else
    GET_FUNCTION_ADDR(ptr_to_ConvolvePdfs_general);
  initialise(paramIndices);
  setIntegrationConstants(intvar->lowerlimit,intvar->upperlimit,intvar->numbins,obs->lowerlimit,obs->upperlimit,obs->numbins);
} 

__host__ void GeneralConvolutionPdf::setIntegrationConstants (fptype intvar_lo, fptype intvar_hi, int intvar_numbins, fptype obs_lo, fptype obs_hi, int obs_numbins) {
  if (!host_iConsts) {
    host_iConsts = new fptype[6]; 
    gooMalloc((void**) &dev_iConsts, 6*sizeof(fptype)); 
  }
  host_iConsts[0] = obs_lo;
  host_iConsts[1] = (obs_hi-obs_lo)/obs_numbins;
  host_iConsts[2] = (intvar_hi-intvar_lo)/intvar_numbins;
  MEMCPY_TO_SYMBOL(functorConstants, host_iConsts, 3*sizeof(fptype), cIndex*sizeof(fptype), cudaMemcpyHostToDevice); 
  if (modelWorkSpace) {
    delete modelWorkSpace;
    delete resolWorkSpace;
  }
  // int M(v)*R(v,x) dv
  // for the Model, it takes v from intvar_lo to intvar_hi

  host_iConsts[0] = obs_lo;
  host_iConsts[1] = obs_hi;
  host_iConsts[2] = obs_numbins;
  host_iConsts[3] = intvar_lo;
  host_iConsts[4] = intvar_hi;
  host_iConsts[5] = intvar_numbins;
  MEMCPY(dev_iConsts, host_iConsts, 6*sizeof(fptype), cudaMemcpyHostToDevice); 
  // build a metric: R[intvar_numbins][obs_numbins]

  fptype* dev_address[1];
  modelWorkSpace = new DEVICE_VECTOR<fptype>(intvar_numbins);
  dev_address[0] = thrust::raw_pointer_cast(modelWorkSpace->data());
  MEMCPY_TO_SYMBOL(dev_modWorkSpace_general, dev_address, sizeof(fptype*), workSpaceIndex*sizeof(fptype*), cudaMemcpyHostToDevice); 
  resolWorkSpace = new DEVICE_VECTOR<fptype>(intvar_numbins*obs_numbins);
  dev_address[0] = thrust::raw_pointer_cast(resolWorkSpace->data());
  MEMCPY_TO_SYMBOL(dev_resWorkSpace_general, dev_address, sizeof(fptype*), workSpaceIndex*sizeof(fptype*), cudaMemcpyHostToDevice); 
}

__host__ fptype GeneralConvolutionPdf::normalise () const {
  // First set normalisation factors to one so we can evaluate convolution without getting zeroes
  recursiveSetNormalisation(fptype(1.0)); 
//  MEMCPY_TO_SYMBOL(normalisationFactors, host_normalisation, totalParams*sizeof(fptype), 0, cudaMemcpyHostToDevice); 

  // Next recalculate functions at each point, in preparation for convolution integral
  thrust::counting_iterator<int> binIndex(0); 

  if (model->parametersChanged()) {
    model->normalise();  // this is needed for the GeneralConvolution
    thrust::constant_iterator<fptype*> model_startendN(dev_iConsts+3); 
    thrust::constant_iterator<int> model_eventSize(1);
    // Calculate model function at every point in integration space
    BinnedMetricTaker modalor(model, getMetricPointer("ptr_to_Eval")); 
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(binIndex, model_eventSize, model_startendN)),
        thrust::make_zip_iterator(thrust::make_tuple(binIndex + modelWorkSpace->size(), model_eventSize, model_startendN)),
        modelWorkSpace->begin(), 
        modalor);
    SYNCH(); 
    model->storeParameters();
  }

  if (resolution->parametersChanged()) {
    resolution->normalise();  // this is needed for the GeneralConvolution
    thrust::constant_iterator<int> res_eventSize(2);
    thrust::constant_iterator<fptype*> res_startendN(dev_iConsts); 
    BinnedMetricTaker resalor(resolution, getMetricPointer("ptr_to_Eval")); 
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(binIndex, res_eventSize, res_startendN)),
        thrust::make_zip_iterator(thrust::make_tuple(binIndex + resolWorkSpace->size(), res_eventSize, res_startendN)),
        resolWorkSpace->begin(), 
        resalor);
    resolution->storeParameters(); 
  }

  //SYNCH(); 

  // Then return usual integral
  //  fptype ret = GooPdf::normalise();
  host_normalisation[parameters] = 1.0;
  return 1; 
}
