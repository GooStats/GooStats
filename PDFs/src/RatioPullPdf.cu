/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "RatioPullPdf.h"

#define M_PI_L 3.141592653589793238462643383279502884L
__host__ RatioPullPdf::RatioPullPdf(std::string n, Variable* var1, Variable* var2, fptype m,fptype s) : 
  DataPdf(nullptr, n),
  index_v1 (registerParameter(var1)),
  index_v2 (registerParameter(var2)),
  data(m),
  sigma(s)
{}


__host__ fptype RatioPullPdf::calculateNLL () const {
  const fptype par = host_params[index_v1]/host_params[index_v2];
  fptype ret = pow((par-data)/sigma,2);
  if(!IsChisquareFit()) ret = ret/2;
#ifdef NLL_CHECK
  printf("log(L) %.12le pull (%.2lf/%.2lf,%.2lf,%.2lf)\n",ret, host_params[index_v1],host_params[index_v2],data,sigma);
#endif
  return ret;
}

#include "TRandom.h"
std::unique_ptr<fptype []> RatioPullPdf::fill_random() {
  std::unique_ptr<fptype[]> h_ptr(new fptype[1]);
  data = gRandom->Gaus(host_params[index_v1]/host_params[index_v2],sigma);
  h_ptr[0] = data;
  return h_ptr;
}
std::unique_ptr<fptype []> RatioPullPdf::fill_Asimov() {
  std::unique_ptr<fptype[]> h_ptr(new fptype[1]);
  data = host_params[index_v1]/host_params[index_v2];
  h_ptr[0] = data;
  return h_ptr;
}
void RatioPullPdf::cache() {
  data_backup = data;
}
void RatioPullPdf::restore() {
  data = data_backup;
}
