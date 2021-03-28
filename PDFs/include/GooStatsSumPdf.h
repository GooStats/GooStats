/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef GooStatsSumPdf_H
#define GooStatsSumPdf_H

#include "goofit/PDFs/SumPdf.h"

class GooStatsSumPdf : public SumPdf {
public:

  GooStatsSumPdf (std::string n, const fptype norm_,const std::vector<Variable*> &weights, const std::vector<Variable*> &sysi,Variable *sys,const std::vector<PdfBase*> &comps,Variable *npe) : SumPdf(n,norm_,weights,sysi,sys,comps,npe) {}
  GooStatsSumPdf (std::string n, const fptype norm_,const std::vector<Variable*> &weights, const std::vector<PdfBase*> &comps, Variable *npe) :
SumPdf(n,norm_,weights,comps,npe) {}
  GooStatsSumPdf (std::string n, const std::vector<PdfBase*> &comps, const std::vector<const BinnedDataSet*> &mask,Variable *npe) :
SumPdf(n,comps,mask,npe) {}
  GooStatsSumPdf (std::string n, const std::vector<PdfBase*> &comps, Variable *npe) :
SumPdf(n,comps,npe) {}

protected:
#ifdef NLL_CHECK
  __host__ double sumOfNll (int numVars) const final;
  friend class DarkNoiseConvolutionPdf;
#endif
};

#endif
