/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef PoissonPullPdf_H
#define PoissonPullPdf_H

#include "DataPdf.h"

/*! \class PoissonPullPdf
 *  \brief The simplest Gaussian pull
 */
class PoissonPullPdf : public DataPdf {
  public:
    PoissonPullPdf(std::string n, Variable* var,Variable *eff,
	fptype mt/* mt is the exposure of the subsidiary exp. */,
	fptype k,fptype b=0);

    __host__ virtual fptype normalise () const{return 1;}

    __host__ fptype calculateNLL() const;
    std::unique_ptr<fptype []> fill_random() final;
    std::unique_ptr<fptype []> fill_Asimov() final;
    void cache() final;
    void restore() final;

  private:
    const int index;
    const int index_e;
    fptype data;
    fptype data_backup = -99;
    const fptype bkg;
    const fptype masstime;
};

#endif
