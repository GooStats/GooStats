/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef PRODUCT_PDF_HH
#define PRODUCT_PDF_HH

#include <map>
#include "goofit/PDFs/GooPdf.h"
template<class T>
class DumperPdf;
class SumLikelihoodPdf;
class BinnedDataSet;

class ProductPdf : public GooPdf {
public:

  ProductPdf (std::string n, const std::vector<PdfBase*> &comps, Variable *npe);
  __host__ virtual fptype normalise () const;

protected:
  void set_startstep(fptype norm);
  void register_components(const std::vector<PdfBase*> &comps,int N);
  std::vector<unsigned int> pindices;
  fptype* dev_iConsts; 
  int workSpaceIndex;
  fptype norm;
  mutable bool m_updated;
};

#endif
