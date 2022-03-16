/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef IBDPdf_H
#define IBDPdf_H

#include "goofit/PDFs/GooPdf.h"

class IBDPdf : public GooPdf {
 public:
  IBDPdf(
      std::string n,
      Variable *eNeu);  // ePos dos not include the energy of two gammas. They will be included by the response function
  __host__ virtual fptype normalise() const;
};

#endif
