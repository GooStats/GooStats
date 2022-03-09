/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef ExpPullPdf_H
#define ExpPullPdf_H

#include "DataPdf.h"

/*! \class ExpPullPdf
 *  \brief The exponential pull
 *  interpretate it as a measurement.
 */
class ExpPullPdf : public DataPdf {
 public:
  ExpPullPdf(std::string n,
             Variable* var,
             fptype ul /* upper limit */,
             fptype cl /* corrsponding confidence level, (0,1) */);

  __host__ virtual fptype normalise() const { return 1; }

  __host__ fptype calculateNLL() const;
  std::unique_ptr<fptype[]> fill_random() final;
  std::unique_ptr<fptype[]> fill_Asimov() final;
  void cache() final;
  void restore() final;

 private:
  const int index;
  fptype data;
  fptype data_backup = -99;
};

#endif
