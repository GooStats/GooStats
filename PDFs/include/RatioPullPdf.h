/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef RATIO_PULL_PDF_HH
#define RATIO_PULL_PDF_HH

#include "goofit/PDFs/GooPdf.h"

class RatioPullPdf : public GooPdf {
  public:
    RatioPullPdf(std::string n, Variable* var1, Variable* var2, double m,double s);

    __host__ virtual fptype normalise () const{return 1;}

    __host__ double calculateNLL() const;

  private:
    const int index_v1, index_v2;
    const double mean;
    const double sigma;
};

#endif
