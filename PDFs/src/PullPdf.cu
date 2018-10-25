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
  DataPdf(nullptr, n),
  index (registerParameter(var)),
  data(m),
  sigma(s),
  masstime(mt)
{
  assert(var);
}


__host__ double PullPdf::calculateNLL () const {
  double ret = pow((host_params[index]-data)/sigma,2);
  // Stirling's approximation: lgamma(n+1) = 0.5*log(2*pi)+(n+0.5)*log(n)-n+O(1/n);
  if(!IsChisquareFit()) ret = ret/2+0.5*log(2*M_PI_L*sigma*masstime*sigma*masstime);
#ifdef NLL_CHECK
  printf("log(L) %.12le pull d %.12le mu %.12le sigma %.12le MT %.12le\n",ret, data,host_params[index],sigma,masstime);
#endif
  return ret;
}

#include "TRandom.h"
std::unique_ptr<fptype []> PullPdf::fill_random() {
  std::unique_ptr<fptype[]> h_ptr(new fptype[1]);
  data = gRandom->Gaus(host_params[index],sigma);
  h_ptr[0] = data;
  return h_ptr;
}
std::unique_ptr<fptype []> PullPdf::fill_Asimov() {
  std::unique_ptr<fptype[]> h_ptr(new fptype[1]);
  data = host_params[index];
  h_ptr[0] = data;
  return h_ptr;
}
void PullPdf::cache() {
  data_backup = data;
}
void PullPdf::restore() {
  data = data_backup;
}
