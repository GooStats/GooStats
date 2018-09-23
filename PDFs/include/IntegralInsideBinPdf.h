/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef INTEGRAL_INSIDE_BIN_PDF_H
#define INTEGRAL_INSIDE_BIN_PDF_H

#include "goofit/PDFs/GooPdf.h"

class IntegralInsideBinPdf : public GooPdf {
  public:
    IntegralInsideBinPdf (std::string n, Variable *x,unsigned int,GooPdf *);
    __host__ virtual fptype normalise () const;
  private:
    unsigned int N;
  fptype* dev_iConsts; 
};


#endif
