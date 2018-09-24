/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "ResponseFunctionPdf.h"
template<> EXEC_TARGET fptype GetNL<ResponseFunctionPdf::NL::Mach4>(fptype *evt,fptype *p,unsigned int *indices) {
  const fptype Eraw = (evt[RO_CACHE(indices[RO_CACHE(indices[0])+3])]); /* in MeV */
  /* parameters follows the exact order of "registerParameter" */ 
  /* Non-linearity parameters */ 
  const fptype ly = RO_CACHE(p[RO_CACHE(indices[_NL_index+0])]);  /* light yield, npe/MeV */ 
  const fptype qc1_ = RO_CACHE(p[RO_CACHE(indices[_NL_index+1])]); 
  const fptype qc2_ = RO_CACHE(p[RO_CACHE(indices[_NL_index+2])]); 
  return (1-POW(1+Eraw/qc1_,qc2_))*ly*Eraw; 
}
template<> EXEC_TARGET fptype GetNL<ResponseFunctionPdf::NL::Echidna>(fptype *evt,fptype *p,unsigned int *indices) {
  const fptype Eraw = (evt[RO_CACHE(indices[RO_CACHE(indices[0])+3])]); // in MeV
  /* parameters follows the exact order of "registerParameter" */ 
  /* Non-linearity parameters */ 
  const fptype ly = RO_CACHE(p[RO_CACHE(indices[_NL_index+0])]);  /* light yield, npe/MeV */ 
  const fptype fCher = RO_CACHE(p[RO_CACHE(indices[_NL_index+1])]); 
  const fptype q1 = RO_CACHE(p[RO_CACHE(indices[_NL_index+2])]); 
  const fptype q2 = RO_CACHE(p[RO_CACHE(indices[_NL_index+3])]); 
  const fptype q3 = RO_CACHE(p[RO_CACHE(indices[_NL_index+4])]); 
  const fptype q4 = RO_CACHE(p[RO_CACHE(indices[_NL_index+5])]); 
  const fptype q5 = RO_CACHE(p[RO_CACHE(indices[_NL_index+6])]); 
  const fptype xQch = LOG(Eraw); // 1 MeV ~ xQch = 0
  const fptype funQch = (q1+q2*xQch+q3*xQch*xQch)/(1+q4*xQch+q5*xQch*xQch); /* qch part */
  const fptype CherTh = RO_CACHE(p[RO_CACHE(indices[_NL_index+7])]); // Cherenkov threshold, in MeV
  const fptype A0 = RO_CACHE(p[RO_CACHE(indices[_NL_index+8])]); 
  const fptype A1 = RO_CACHE(p[RO_CACHE(indices[_NL_index+9])]); 
  const fptype A2 = RO_CACHE(p[RO_CACHE(indices[_NL_index+10])]); 
  const fptype A3 = RO_CACHE(p[RO_CACHE(indices[_NL_index+11])]); 
  const fptype A4 = RO_CACHE(p[RO_CACHE(indices[_NL_index+12])]); 
  const fptype xCher = LOG(1+Eraw/CherTh);
  const fptype funCher = ((CherTh==0)||(Eraw<CherTh))?0:((A0+A1*xCher+A2*xCher*xCher+A4*xCher*xCher*xCher)*(1+A3*Eraw));
  return Eraw*ly*(funQch+fCher*funCher);
}
template<> EXEC_TARGET fptype GetNL<ResponseFunctionPdf::NL::expPar>(fptype *evt,fptype *p,unsigned int *indices) {
  const fptype Eraw = (evt[RO_CACHE(indices[RO_CACHE(indices[0])+3])]); // in MeV
  /* parameters follows the exact order of "registerParameter" */ 
  /* Non-linearity parameters */ 
  const fptype ly = RO_CACHE(p[RO_CACHE(indices[_NL_index+0])]);  /* light yield, npe/keV */ 
  const fptype b = RO_CACHE(p[RO_CACHE(indices[_NL_index+1])]); 
  const fptype c = RO_CACHE(p[RO_CACHE(indices[_NL_index+2])]); 
  const fptype e = RO_CACHE(p[RO_CACHE(indices[_NL_index+3])]); 
  const fptype f = RO_CACHE(p[RO_CACHE(indices[_NL_index+4])]); 
  const fptype X = LOG(Eraw);
  return (1+X*b+X*X*c)/(1+X*e+X*X*f)*ly*Eraw;
}

template<> EXEC_TARGET fptype GetMean<ResponseFunctionPdf::Mean::normal>(fptype mu,fptype *,unsigned int *) {
  return mu;
}
template<> EXEC_TARGET fptype GetMean<ResponseFunctionPdf::Mean::peak>(fptype ,fptype *p,unsigned int *indices) {
  const int _NL_size = RO_CACHE(indices[1]);  /* number of non-linearity parameters */ 
  const int _Res_size = RO_CACHE(indices[_NL_index+_NL_size]);  /* number of resolution parameters */ 
  const int _peakE_index = _NL_index+_NL_size+1+_Res_size+1;  /* internal index for the peak E parameter */ 
  return RO_CACHE(p[RO_CACHE(indices[_peakE_index+0])]);  
}
template<> EXEC_TARGET fptype GetMean<ResponseFunctionPdf::Mean::shifted>(fptype mu,fptype *p,unsigned int *indices) {
  const int _NL_size = RO_CACHE(indices[1]);  /* number of non-linearity parameters */ 
  const int _Res_size = RO_CACHE(indices[_NL_index+_NL_size]);  /* number of resolution parameters */ 
  const int _shiftE_index = _NL_index+_NL_size+1+_Res_size+1;  /* internal index for the shift E parameter */ 
  return mu+RO_CACHE(p[RO_CACHE(indices[_shiftE_index+0])]);  
}
template<> EXEC_TARGET fptype GetVariance<ResponseFunctionPdf::RES::charge>(fptype mu,fptype *p,unsigned int *indices) {
  const int _NL_size = RO_CACHE(indices[1]);  /* number of non-linearity parameters */ 
  const int _Res_index = _NL_index+_NL_size+1;  /* internal index for the first resolution parameter */
  /* Resolution parameters */ 
  const fptype sdn = RO_CACHE(p[RO_CACHE(indices[_Res_index+0])]);  
  const fptype v1 = RO_CACHE(p[RO_CACHE(indices[_Res_index+1])]);  
  const fptype sigmaT = RO_CACHE(p[RO_CACHE(indices[_Res_index+2])]); 
  /* feq : equalization factor introduced to remove loss of channels */ 
  const int _Res_size = RO_CACHE(indices[_NL_index+_NL_size]);  /* number of resolution parameters */ 
  const int _feq_index = _Res_index+_Res_size;  /*  internal index for feq */
  const unsigned int cIndex = RO_CACHE(indices[_feq_index]); 
  const fptype feq = RO_CACHE(functorConstants[cIndex]);
  return sdn*sdn + mu*(1.+v1) *feq + sigmaT*sigmaT*mu*mu;
}
