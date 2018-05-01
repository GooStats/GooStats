/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "PullPdf.h"

#define M_PI_L 3.141592653589793238462643383279502884L
__host__ PullPdf::PullPdf(std::string n, Variable* var, fptype m,fptype s,fptype mt) :
  GooPdf(nullptr, n),
  index (registerParameter(var)),
  mean(m*mt),
  sigma(s*mt),
  masstime(mt)
{}


__host__ double PullPdf::calculateNLL () const {
  const double counts = masstime*host_params[index];
  const double constTerm(IsChisquareFit()?(0.5*log(2*M_PI_L*sigma*sigma)):0);
#ifdef NLL_CHECK
  printf("log(L) %.12le pull\n",(counts-mean)*(counts-mean)/(2*sigma*sigma)+constTerm);
#endif
  return (counts-mean)*(counts-mean)/(2*sigma*sigma)+constTerm;
}

