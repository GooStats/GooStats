#include "goofit/FitControl.h"
MEM_CONSTANT fptype dev_Nll_threshold[1];
MEM_CONSTANT fptype dev_Nll_scaleFactor[1];
BinnedNllScaledFit::BinnedNllScaledFit(double threshold,double scaleFactor) :
  FitControl(true,"ptr_to_ScaledBinAvg")
{
  MEMCPY_TO_SYMBOL(dev_Nll_threshold,&threshold,sizeof(fptype),0,cudaMemcpyHostToDevice);
  MEMCPY_TO_SYMBOL(dev_Nll_scaleFactor,&scaleFactor,sizeof(fptype),0,cudaMemcpyHostToDevice);
}
