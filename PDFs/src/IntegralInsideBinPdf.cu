/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "IntegralInsideBinPdf.h"
#include "SumPdf.h"

EXEC_TARGET fptype device_IntegralInsideBin (fptype* evt, fptype* , unsigned int* indices) { 
  const fptype npe_val = evt[RO_CACHE(indices[2 + RO_CACHE(indices[0])])]; 
  const int cIndex = RO_CACHE(indices[1]); 
  const fptype npe_lo = RO_CACHE(functorConstants[cIndex]);
  const fptype npe_step = RO_CACHE(functorConstants[cIndex+1]);
  const int npe_bin = (int) FLOOR((npe_val-npe_lo)/npe_step); // no problem with FLOOR: start from 0.5, which corresponse to bin=0
  const int workSpaceIndex = RO_CACHE(indices[2]); // ok
  const int Nsub = RO_CACHE(indices[3]); // ok

  fptype sum = 0;
  for(int i = 0;i<Nsub;++i) {
    sum += dev_componentWorkSpace[workSpaceIndex][npe_bin*Nsub+i];
#ifdef RPF_CHECK
    if(THREADIDX==0)
      printf("%d %lf %d / %d (%d/%d) <- %le , %le\n",
	  THREADIDX, npe_val, npe_bin,npe_bin*Nsub+i, i,Nsub,dev_componentWorkSpace[workSpaceIndex][npe_bin*Nsub+i], sum);
#endif
  }
  return sum/Nsub;
}

MEM_DEVICE device_function_ptr ptr_to_IntegralInsideBin = device_IntegralInsideBin;

  IntegralInsideBinPdf::IntegralInsideBinPdf (std::string n, Variable* x, unsigned int scale,GooPdf *pdf)
: GooPdf(x, n), N(x->numbins*scale)
{
  // pdf
  components.push_back(pdf);
  std::vector<unsigned int> paramIndices;
  paramIndices.push_back(registerConstants(2));
  assert(pdf);
  const int workSpaceIndex = SumPdf::registerFunc(pdf);
  assert(workSpaceIndex<NPDFSIZE_SumPdf);
  componentWorkSpace[workSpaceIndex] = new DEVICE_VECTOR<fptype>(N);
  fptype *dev_address = thrust::raw_pointer_cast(componentWorkSpace[workSpaceIndex]->data());
  MEMCPY_TO_SYMBOL(dev_componentWorkSpace, &dev_address, sizeof(fptype*), workSpaceIndex*sizeof(fptype*), cudaMemcpyHostToDevice); 
  paramIndices.push_back(workSpaceIndex); // 1
  // scale
  paramIndices.push_back(scale); // 2
  // integral
  assert(x);
  fptype host_iConsts[3];
  host_iConsts[0] = x->lowerlimit;
  host_iConsts[1] = (x->upperlimit-x->lowerlimit)/x->numbins;
  MEMCPY_TO_SYMBOL(functorConstants, host_iConsts, 2*sizeof(fptype), cIndex*sizeof(fptype), cudaMemcpyHostToDevice);  // cIndex is a member derived from PdfBase and is set inside registerConstants method
  gooMalloc((void**) &dev_iConsts, 3*sizeof(fptype)); 
  host_iConsts[0] = x->lowerlimit;
  host_iConsts[1] = x->upperlimit;
  host_iConsts[2] = x->numbins*scale;
  MEMCPY(dev_iConsts, host_iConsts, 3*sizeof(fptype), cudaMemcpyHostToDevice); 
  GET_FUNCTION_ADDR(ptr_to_IntegralInsideBin);
  initialise(paramIndices);
} 

__host__ fptype IntegralInsideBinPdf::normalise () const {
  host_normalisation[parameters] = 1.0; 
  thrust::counting_iterator<int> binIndex(0); 

  GooPdf *pdf = static_cast<GooPdf*>(components.at(0));
  const int workSpaceIndex = SumPdf::registerFunc(pdf);
  if (pdf->parametersChanged()) {
    pdf->normalise();  // this is needed for the GeneralConvolution
    thrust::constant_iterator<fptype*> startendstep(dev_iConsts); // 3*fptype lo, hi and step for npe
    thrust::constant_iterator<int> eventSize(1);
    // Calculate pdf function at every point in integration space
    BinnedMetricTaker modalor(pdf, getMetricPointer("ptr_to_Eval")); 
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(binIndex, eventSize, startendstep)),
	thrust::make_zip_iterator(thrust::make_tuple(binIndex + N, eventSize, startendstep)),
	componentWorkSpace[workSpaceIndex]->begin(),
	modalor);
    SYNCH(); 
    pdf->storeParameters();
  }
  return 1; 
}
