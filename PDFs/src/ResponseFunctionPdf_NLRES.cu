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
  const fptype Eraw = (evt[RO_CACHE(indices[RO_CACHE(indices[0])+3])]); 
  /* parameters follows the exact order of "registerParameter" */ 
  /* Non-linearity parameters */ 
  const fptype ly = RO_CACHE(p[RO_CACHE(indices[_NL_index+0])]);  /* light yield, npe/keV */ 
  const fptype qc1_ = RO_CACHE(p[RO_CACHE(indices[_NL_index+1])]); 
  const fptype qc2_ = RO_CACHE(p[RO_CACHE(indices[_NL_index+2])]); 
  return (1-POW(1+Eraw*1000/qc1_,qc2_))*ly*Eraw; /* Eraw in MeV */ 
}
template<> EXEC_TARGET fptype GetNL<ResponseFunctionPdf::NL::expPar>(fptype *evt,fptype *p,unsigned int *indices) {
  const fptype Eraw = (evt[RO_CACHE(indices[RO_CACHE(indices[0])+3])]); 
  /* parameters follows the exact order of "registerParameter" */ 
  /* Non-linearity parameters */ 
  const fptype ly = RO_CACHE(p[RO_CACHE(indices[_NL_index+0])]);  /* light yield, npe/keV */ 
  const fptype b = RO_CACHE(p[RO_CACHE(indices[_NL_index+1])]); 
  const fptype c = RO_CACHE(p[RO_CACHE(indices[_NL_index+2])]); 
  const fptype e = RO_CACHE(p[RO_CACHE(indices[_NL_index+3])]); 
  const fptype f = RO_CACHE(p[RO_CACHE(indices[_NL_index+4])]); 
  const fptype X = LOG(Eraw);
  return (1+X*b+X*X*c)/(1+X*e+X*X+f)*ly*Eraw;
}

template<> EXEC_TARGET fptype GetMean<ResponseFunctionPdf::Mean::normal>(fptype mu,fptype *,unsigned int *) {
  return mu;
}
template<> EXEC_TARGET fptype GetMean<ResponseFunctionPdf::Mean::peak>(fptype ,fptype *p,unsigned int *indices) {
  const int _NL_size = RO_CACHE(indices[1]);  /* light yield, npe/keV */ 
  const int _res_size = RO_CACHE(indices[_NL_index+_NL_size]);  /* light yield, npe/keV */ 
  const int _peakE_index = _NL_index+_NL_size+1+_res_size+1;  /* light yield, npe/keV */ 
  return RO_CACHE(p[RO_CACHE(indices[_peakE_index+0])]);  
}
template<> EXEC_TARGET fptype GetMean<ResponseFunctionPdf::Mean::shifted>(fptype mu,fptype *p,unsigned int *indices) {
  const int _NL_size = RO_CACHE(indices[1]);  /* light yield, npe/keV */ 
  const int _Res_size = RO_CACHE(indices[_NL_index+_NL_size]);  /* light yield, npe/keV */ 
  const int _shiftE_index = _NL_index+_NL_size+1+_Res_size+1;  /* light yield, npe/keV */ 
  return mu+RO_CACHE(p[RO_CACHE(indices[_shiftE_index+0])]);  
}
template<> EXEC_TARGET fptype GetVariance<ResponseFunctionPdf::RES::charge>(fptype mu,fptype *p,unsigned int *indices) {
  const int _NL_size = RO_CACHE(indices[1]);  /* light yield, npe/keV */ 
  const int _Res_index = _NL_index+_NL_size+1;  /* light yield, npe/keV */ 
  /* Resolution parameters */ 
  const fptype v1 = RO_CACHE(p[RO_CACHE(indices[_Res_index+0])]);  
  const fptype sigmaT = RO_CACHE(p[RO_CACHE(indices[_Res_index+1])]); 
  /* feq : equalization factor introduced to remove loss of channels */ 
  const int _Res_size = RO_CACHE(indices[_NL_index+_NL_size]);  /* light yield, npe/keV */ 
  const int _feq_index = _Res_index+_Res_size;  /* light yield, npe/keV */ 
  const unsigned int cIndex = RO_CACHE(indices[_feq_index]); 
  const fptype feq = RO_CACHE(functorConstants[cIndex]);
  return mu*(1.+v1) *feq + sigmaT*sigmaT*mu*mu;
}
