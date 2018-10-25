/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "PoissonPullPdf.h"

#define M_PI_L 3.141592653589793238462643383279502884L
__host__ PoissonPullPdf::PoissonPullPdf(std::string n, Variable* var, Variable *eff,
    fptype mt,fptype k,fptype b) :
  DataPdf(nullptr, n),
  index (registerParameter(var)),
  index_e (registerParameter(eff)),
  data(k),
  bkg(b),
  masstime(mt)
{}


__host__ double PoissonPullPdf::calculateNLL () const {
  const double var = host_params[index];
  const double eff = host_params[index_e];
  int k = data;
  double lambda = var*masstime*eff+bkg;
  double ret = -(k*log(lambda)-lgamma(k+1.)-lambda);
  // Stirling's approximation: lgamma(n+1) = 0.5*log(2*pi)+(n+0.5)*log(n)-n+O(1/n);
  if(IsChisquareFit()) ret = ret*2;
#ifdef NLL_CHECK
  printf("log(L) %.12le pull chisquare? %s\n",ret, IsChisquareFit()?"yes":"no");
#endif
  return ret;
}

#include "TRandom.h"
std::unique_ptr<fptype []> PoissonPullPdf::fill_random() {
  std::unique_ptr<fptype[]> h_ptr(new fptype[1]);
  data = gRandom->Poisson(host_params[index]*masstime+bkg);
  h_ptr[0] = data;
  return h_ptr;
}
std::unique_ptr<fptype []> PoissonPullPdf::fill_Asimov() {
  std::unique_ptr<fptype[]> h_ptr(new fptype[1]);
  data = host_params[index]*masstime+bkg;
  h_ptr[0] = data;
  return h_ptr;
}
void PoissonPullPdf::cache() {
  data_backup = data;
}
void PoissonPullPdf::restore() {
  data = data_backup;
}
