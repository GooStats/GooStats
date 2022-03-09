/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "ExpPullPdf.h"

#define M_PI_L 3.141592653589793238462643383279502884L
__host__ ExpPullPdf::ExpPullPdf(std::string n, Variable* var, fptype ul, fptype cl)
    : DataPdf(nullptr, n), index(registerParameter(var)), data(-ul / log(cl)) {
  assert(cl > 0 && cl < 1 && ul > 0);
}

__host__ double ExpPullPdf::calculateNLL() const {
  const double mu = host_params[index];
  double ret = data / mu + log(mu);
  if (IsChisquareFit())
    ret = ret * 2;
#ifdef NLL_CHECK
  printf("log(L) %.12le pull d %.12le mu %.12le\n", ret, data, mu);
#endif
  return ret;
}

#include "TRandom.h"
std::unique_ptr<fptype[]> ExpPullPdf::fill_random() {
  std::unique_ptr<fptype[]> h_ptr(new fptype[1]);
  data = gRandom->Exp(host_params[index]);
  h_ptr[0] = data;
  return h_ptr;
}
std::unique_ptr<fptype[]> ExpPullPdf::fill_Asimov() {
  std::unique_ptr<fptype[]> h_ptr(new fptype[1]);
  data = host_params[index];
  h_ptr[0] = data;
  return h_ptr;
}
void ExpPullPdf::cache() { data_backup = data; }
void ExpPullPdf::restore() { data = data_backup; }
