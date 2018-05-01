/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef PULLPDF_HH
#define PULLPDF_HH

#include "goofit/PDFs/GooPdf.h"

class PullPdf : public GooPdf {
  public:
    PullPdf(std::string n, Variable* var, fptype m,fptype s,fptype mt);

    __host__ virtual fptype normalise () const{return 1;}

    __host__ fptype calculateNLL() const;

  private:
    const int index;
    const fptype mean;
    const fptype sigma;
    const fptype masstime;
};

#endif
