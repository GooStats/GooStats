/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef DataPdf_H
#define DataPdf_H

#include "goofit/PDFs/GooPdf.h"

class DataPdf : public GooPdf {
 public:
  DataPdf(Variable *var, std::string n) : GooPdf(var, n){};
  virtual std::unique_ptr<fptype[]> fill_random() = 0;
  virtual std::unique_ptr<fptype[]> fill_Asimov() = 0;
  virtual void cache() = 0;
  virtual void restore() = 0;
  virtual int NDF() = 0;
  virtual int Nfree() = 0;
};

#endif
