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

#include "DataPdf.h"

class PullPdf : public DataPdf {
  public:
    PullPdf(std::string n, Variable* var, fptype m,fptype s,fptype mt);

    __host__ virtual fptype normalise () const{return 1;}

    __host__ fptype calculateNLL() const;
    std::unique_ptr<fptype []> fill_random() final;
    std::unique_ptr<fptype []> fill_Asimov() final;
  void cache();
  void restore();

  private:
    const int index;
    fptype mean;
    fptype mean_backup = -99;
    const fptype sigma;
    const fptype masstime;
};

#endif
