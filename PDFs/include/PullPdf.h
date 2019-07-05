/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef PullPdf_H
#define PullPdf_H

#include "DataPdf.h"

/*! \class PullPdf
 *  \brief The simplest Gaussian pull
 */
class PullPdf : public DataPdf {
  public:
    PullPdf(std::string n, Variable* var,
	fptype m,fptype s,
	fptype mt/* mt is the exposure of the subsidiary exp. */,bool half_ = false); // minus infinity 

    __host__ virtual fptype normalise () const{return 1;}

    __host__ fptype calculateNLL() const;
    std::unique_ptr<fptype []> fill_random() final;
    std::unique_ptr<fptype []> fill_Asimov() final;
    void cache() final;
    void restore() final;
    int NDF() final { return 0; }
    int Nfree() final { return 1; }

  private:
    const int index;
    fptype data;
    fptype data_backup = -99;
    const fptype sigma;
    const fptype masstime;
    const bool half;
};

#endif
