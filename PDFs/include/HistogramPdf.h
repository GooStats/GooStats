/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef SHIFTED_HISTOGRAM_PDF_HH
#define SHIFTED_HISTOGRAM_PDF_HH

#include "goofit/PDFs/GooPdf.h"
#include "goofit/BinnedDataSet.h"

class HistogramPdf : public GooPdf {
public:
  HistogramPdf (std::string n, BinnedDataSet* x,Variable *scale = nullptr,Variable *shift = nullptr,bool alreadyNormalized = false);
  __host__ virtual fptype normalise () const;
  __host__ void extractHistogram (thrust::host_vector<fptype>& host_hist) {host_hist = *dev_base_histogram;}
  __host__ void copyHistogramToDevice (thrust::host_vector<fptype>& host_histogram);

private:
  DEVICE_VECTOR<fptype>* dev_base_histogram; 
//  DEVICE_VECTOR<fptype>* dev_smoothed_histogram; 
  fptype* host_constants;

  static unsigned int totalHistograms; 
};

#endif
