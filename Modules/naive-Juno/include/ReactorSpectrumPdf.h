/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef ReactorSpectrumPdf_H
#define ReactorSpectrumPdf_H

#include "goofit/PDFs/GooPdf.h"

class ReactorSpectrumPdf : public GooPdf {
  public:
    ReactorSpectrumPdf (std::string n, Variable *x, const std::vector<Variable *> &fractions,
	const std::vector<double> &coefficients);
    __host__ virtual fptype normalise () const;
};

#endif
