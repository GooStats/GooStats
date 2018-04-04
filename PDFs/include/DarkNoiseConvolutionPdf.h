/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef DARKNOISECONVOLUTION_PDF_HH
#define DARKNOISECONVOLUTION_PDF_HH
#include "goofit/PDFs/GooPdf.h"

class DarkNoiseConvolutionPdf: public GooPdf {
  public:
    DarkNoiseConvolutionPdf(std::string n,Variable *npe,GooPdf *rawPdf_,BinnedDataSet *dn_histo);
  private:
#ifdef NLL_CHECK
  __host__ fptype sumOfNll (int numVars) const;
#endif
    __host__ void copyHistogramToDevice (BinnedDataSet *dn_histo);
    __host__ void setIntegrationConstants();
    __host__ virtual fptype normalise () const;
    GooPdf* rawPdf;
    fptype* dev_iConsts; 
    int id;
    DEVICE_VECTOR<fptype>* modelWorkSpace;
    int npe_lo;
    int npe_hi;
    int dn_max;
};
#endif
