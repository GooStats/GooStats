/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef RatioPullPdf_H
#define RatioPullPdf_H

#include "DataPdf.h"

/*! \class RatioPullPdf
 *  \brief The gaussian pull on the ratio of two variables
 */
class RatioPullPdf : public DataPdf {
 public:
  RatioPullPdf(std::string n, Variable* var1, Variable* var2, fptype m, fptype s);

  __host__ virtual fptype normalise() const { return 1; }

  __host__ fptype calculateNLL() const;
  std::unique_ptr<fptype[]> fill_random() final;
  std::unique_ptr<fptype[]> fill_Asimov() final;
  void cache() final;
  void restore() final;
  int NDF() final { return 0; }
  int Nfree() final { return 1; }

 private:
  const int index_v1, index_v2;
  fptype data;
  fptype data_backup = -99;
  const fptype sigma;
};

#endif
