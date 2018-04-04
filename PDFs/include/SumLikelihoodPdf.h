/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef SUM_LIKELIHOOD_PDF_HH
#define SUM_LIKELIHOOD_PDF_HH

#include <map>
#include "goofit/PDFs/GooPdf.h"
class Variable;

class SumLikelihoodPdf : public GooPdf {
public:

  SumLikelihoodPdf (std::string n, const std::vector<PdfBase*> &comps);
  __host__ virtual fptype normalise () const;
  const std::vector<PdfBase*> &Components() const { return components; }
  void fill_random();

private:
  __host__ void setData(BinnedDataSet *data);
  __host__ void setData(UnbinnedDataSet *data);

protected:
  __host__ virtual double sumOfNll (int numVars) const;
};

#endif
