/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef RangedExpPdf_H
#define RangedExpPdf_H

#include "goofit/PDFs/GooPdf.h"

class RangedExpPdf : public GooPdf {
public:

  RangedExpPdf (std::string n, Variable *npe,Variable* p0, Variable* p1, fptype x_L, fptype x_H, bool inversed);
  __host__ virtual fptype normalise () const;

protected:
  std::vector<unsigned int> pindices;
};

#endif
