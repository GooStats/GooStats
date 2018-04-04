/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef GENERALCONVOLVE_PDF_HH
#define GENERALCONVOLVE_PDF_HH

#include "goofit/PDFs/GooPdf.h"

class GeneralConvolutionPdf : public GooPdf {
public:

  GeneralConvolutionPdf (std::string n, Variable* x, Variable* intvar, GooPdf* model, GooPdf* resolution, bool syn_loading = false); 
  __host__ virtual fptype normalise () const;
  __host__ void setIntegrationConstants (fptype intvar_lo, fptype intvar_hi, int intvar_numbins, fptype obs_lo, fptype obs_hi, int obs_numbins);

protected:
  GooPdf* model;
  GooPdf* resolution; 

  fptype* host_iConsts; 
  fptype* dev_iConsts; 
  DEVICE_VECTOR<fptype>* modelWorkSpace;
  DEVICE_VECTOR<fptype>* resolWorkSpace; 
  int workSpaceIndex; 

};


#endif
