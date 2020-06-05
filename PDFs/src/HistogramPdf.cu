/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "HistogramPdf.h"

MEM_CONSTANT fptype* dev_raw_histograms[100]; // Multiple histograms for the case of multiple PDFs
MEM_CONSTANT fptype* dev_quenched_histograms[100]; 
unsigned int HistogramPdf::totalHistograms = 0; 

EXEC_TARGET fptype device_EvalRawHistogram (fptype* evt, fptype* , unsigned int* indices) {
    const fptype lo = RO_CACHE(functorConstants[RO_CACHE(indices[ 2])]);
    const fptype step = RO_CACHE(functorConstants[RO_CACHE(indices[ 2])+1]);
    const int bin = (int) FLOOR((evt[RO_CACHE(indices[RO_CACHE(indices[0]) + 2])]-lo)/step);

  return RO_CACHE((dev_raw_histograms[RO_CACHE(indices[1])])[bin]);
}
EXEC_TARGET fptype device_ScaleShiftEvalRawHistogram (fptype* evt, fptype* p, unsigned int* indices) {
  const unsigned int cIndex = RO_CACHE(indices[2]);
  const fptype lo = RO_CACHE(functorConstants[cIndex]);
  const fptype step = RO_CACHE(functorConstants[cIndex+1]);
  const fptype val_origin = evt[RO_CACHE(indices[RO_CACHE(indices[0]) + 2])];
  const fptype scale = RO_CACHE(p[RO_CACHE(indices[3])]); 
  const fptype shift = RO_CACHE(p[RO_CACHE(indices[4])]); 
  const fptype val = (val_origin-shift)/scale;
  fptype bin_fp = (val-lo)/step;
  int bin_lo = (int) FLOOR(bin_fp);
  const int wid = RO_CACHE(indices[1]);
  if(bin_lo<0) return RO_CACHE((dev_raw_histograms[wid])[0]);
  if(static_cast<unsigned int>(bin_lo)>=RO_CACHE(indices[5])/*bin_max*/) return RO_CACHE((dev_raw_histograms[wid])[bin_lo]);
  const fptype y_lo = RO_CACHE((dev_raw_histograms[wid])[bin_lo]);
  const fptype y_up = RO_CACHE((dev_raw_histograms[wid])[bin_lo+1]);
  return y_lo+(y_up-y_lo)*(bin_fp-bin_lo-0.5);
}

MEM_DEVICE device_function_ptr ptr_to_EvalRawHistogram = device_EvalRawHistogram; 

MEM_DEVICE device_function_ptr ptr_to_ScaleShiftEvalRawHistogram = device_ScaleShiftEvalRawHistogram; 

double CalculateRealNorm(BinnedDataSet *data) {
  Variable *var = *(data->varsBegin());
  unsigned n_pts = var->numbins;
  double de = (var->upperlimit- var->lowerlimit) / (var->numbins); 
  double real_norm_ = 0;

  // if number of points is even, neglect the first interval (for now)
  // in order to use Simpson's Rule
  unsigned start = 1 - n_pts % 2;

  for (unsigned i = start; i < n_pts; i++) {
    unsigned factor = 2;
    if (i == start || i + 1 == n_pts) factor = 1;
    else if ((i - start) % 2 == 1)    factor = 4;

    real_norm_ += factor * data->getBinContent(i);
  }
  real_norm_ /= 3.;

  // take care of first interval if number of points was even
  if (start == 1)
    real_norm_ += 0.5 * (data->getBinContent(0) + data->getBinContent(1));

  real_norm_ *= de;
  return real_norm_;
}
__host__ HistogramPdf::HistogramPdf (std::string n, BinnedDataSet* hist,Variable *scale,Variable *shift,bool alreadyNormalized)
  : GooPdf(*(hist->varsBegin()), n) 
{
  int numVars = hist->numVariables(); 
  if(numVars!=1) 
    abortWithCudaPrintFlush(__FILE__, __LINE__, getName() + " only valid for 1-D histogram", this);
  Variable *energy = *(hist->varsBegin());
  host_constants = new fptype[3]; 

  std::vector<unsigned int> pindices;
  pindices.push_back(totalHistograms);  // 1

  pindices.push_back(registerConstants(2)); // 2
  if(scale&&shift) {
    pindices.push_back(registerParameter(scale));
    pindices.push_back(registerParameter(shift));
    pindices.push_back(hist->getNumBins());
  }

  host_constants[0] = energy->lowerlimit;;
  host_constants[1] = (energy->upperlimit-energy->lowerlimit)/energy->numbins;
  assert(host_constants[1]>0);

  unsigned int numbins = hist->getNumBins(); 
  const fptype norm = alreadyNormalized?1.:CalculateRealNorm(hist);
  assert(norm>0);
  thrust::host_vector<fptype> host_histogram; 
  for (unsigned int i = 0; i < numbins; ++i) {
    fptype curr = hist->getBinContent(i);
    host_histogram.push_back(curr/norm);
  }
  MEMCPY_TO_SYMBOL(functorConstants, host_constants, 2*sizeof(fptype), cIndex*sizeof(fptype), cudaMemcpyHostToDevice); 

  copyHistogramToDevice(host_histogram);

  if(scale&&shift)
    GET_FUNCTION_ADDR(ptr_to_ScaleShiftEvalRawHistogram);
  else
    GET_FUNCTION_ADDR(ptr_to_EvalRawHistogram);
  initialise(pindices); 
}

__host__ void HistogramPdf::copyHistogramToDevice (thrust::host_vector<fptype>& host_histogram) {
  dev_base_histogram = new DEVICE_VECTOR<fptype>(host_histogram);  
  static fptype* dev_address[1];
  dev_address[0] = thrust::raw_pointer_cast(dev_base_histogram->data());
  MEMCPY_TO_SYMBOL(dev_raw_histograms, dev_address, sizeof(fptype*), totalHistograms*sizeof(fptype*), cudaMemcpyHostToDevice); 
  totalHistograms++; 
}

__host__ fptype HistogramPdf::normalise () const {
  host_normalisation[parameters] = 1.0;
  return 1; 
}
