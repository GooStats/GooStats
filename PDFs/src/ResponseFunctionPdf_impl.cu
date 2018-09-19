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

// Nonlinearity model reference: (Mach4 Quenching)
//   [1] R. N. Saldanha, “Precision Measurement of the 7 Be Solar Neutrino Interaction Rate in Borexino,” Princeton University, 2012.
// Resolution model Reference: (Generalized Gamma function)
//   [2] O. J. Smirnov, “An approximation of the ideal scintillation detector line shape with a generalized gamma distribution,” Nucl. Instruments Methods Phys. Res. Sect. A Accel. Spectrometers, Detect. Assoc. Equip., vol. 595, no. 2, pp. 410–418, 2008.
template<ResponseFunctionPdf::NL nl,ResponseFunctionPdf::Mean type,ResponseFunctionPdf::RES res> // type: normal, peak(gamma) or shifted(position)
EXEC_TARGET fptype device_npe_GeneralizedGamma (fptype* evt, fptype* p, unsigned int* indices) {  
  /* Observables */
  const fptype Evis = (evt[RO_CACHE(indices[RO_CACHE(indices[0])+2])]);  

  /* calculate mu */ 
  fptype mu = GetNL<nl>(evt,p,indices);
  mu = GetMean<type>(mu,p,indices);
  /* calculate var */ 
  if(mu<=0.5) return -1e10; 
  const fptype variance = GetVariance<res>(mu,p,indices);
    
  /* Rewrite moments of ideal detector response in terms of charge_mean and variance */ 
  const fptype moment_2 = mu*mu + variance; 
  const fptype moment_4 = (variance*variance*(2.+3./mu) + 4.*mu*mu*(variance - 2.) 
			   + 2.*mu*(6.*variance - 1) 
			   + (variance + mu*mu)*(variance + mu*mu)); 

  /* get alpha and beta from momentums */ 
  const fptype alpha = moment_2*moment_2/(moment_4 - moment_2*moment_2); 
  const fptype beta = alpha/moment_2; 
  const fptype ret = (EXP(LOG(2.) 
			  + alpha*LOG(beta) 
			  + LOG(Evis)*(2.*alpha -1.) 
			  + -beta*Evis*Evis 
			  - ::lgamma(alpha)));  
#ifdef RPF_CHECK
  const fptype Eraw = (evt[RO_CACHE(indices[RO_CACHE(indices[0])+3])]);
  const fptype ly = RO_CACHE(p[RO_CACHE(indices[_NL_index+0])]);  /* light yield, npe/keV */ 
  const fptype qc1_ = RO_CACHE(p[RO_CACHE(indices[_NL_index+1])]); 
  const fptype qc2_ = RO_CACHE(p[RO_CACHE(indices[_NL_index+2])]); 
  const int _NL_size = RO_CACHE(indices[1]);  /* light yield, npe/keV */ 
  const int _Res_index = _NL_index+_NL_size+1;  /* light yield, npe/keV */ 
  const fptype v1 = RO_CACHE(p[RO_CACHE(indices[_Res_index+0])]);  
  const fptype sigmaT = RO_CACHE(p[RO_CACHE(indices[_Res_index+1])]); 
  const int _Res_size = RO_CACHE(indices[_NL_index+_NL_size]);  /* light yield, npe/keV */ 
  const int _feq_index = _Res_index+_Res_size;  /* light yield, npe/keV */ 
  const unsigned int cIndex = RO_CACHE(indices[_feq_index]); 
  const fptype feq = RO_CACHE(functorConstants[cIndex]);
  if(fabs(mu-Evis)<sqrt(variance)*0.2) 
    printf("%d %.1lf <- %.2lf : (%.1lf %.3lf %.3lf) (%.3lf %.1lf) (%.3lf) | (%lf %lf) (%lf %lf) (%lf %lf) -> %lf\n", 
	   THREADIDX, Evis, ly, qc1_, qc2_, v1, sigmaT, feq, mu, variance, moment_2, moment_4, alpha, beta, ret ); 
#endif
  return ret; 
} 
// Modified gaussian:
//   [1] R. N. Saldanha, “Precision Measurement of the 7 Be Solar Neutrino Interaction Rate in Borexino,” Princeton University, 2012.
template<ResponseFunctionPdf::NL nl,ResponseFunctionPdf::Mean type,ResponseFunctionPdf::RES res> // type: normal, peak(gamma) or shifted(position)
EXEC_TARGET fptype device_npe_ModifiedGaussian (fptype* evt, fptype* p, unsigned int* indices) {  
  /* Observables */
  const fptype Evis = (evt[RO_CACHE(indices[RO_CACHE(indices[0])+2])]);  

  /* calculate mu */ 
  fptype mu = GetNL<nl>(evt,p,indices);
  mu = GetMean<type>(mu,p,indices);
  /* calculate var */ 
  if(mu<=0.5) return -1e10; 
  const fptype variance = GetVariance<res>(mu,p,indices);

  /* calculate kappa */ 
  const int _NL_size = RO_CACHE(indices[1]);  /* light yield, npe/keV */ 
  const int _Res_index = _NL_index+_NL_size+1;  /* light yield, npe/keV */ 
  const fptype g2 = RO_CACHE(p[RO_CACHE(indices[_Res_index+2])]); 
  const fptype v1 = RO_CACHE(p[RO_CACHE(indices[_Res_index+0])]);  
  const fptype sigmaT = RO_CACHE(p[RO_CACHE(indices[_Res_index+1])]); 
  const int _Res_size = RO_CACHE(indices[_NL_index+_NL_size]);  /* light yield, npe/keV */ 
  const int _feq_index = _Res_index+_Res_size;  /* light yield, npe/keV */ 
  const unsigned int cIndex = RO_CACHE(indices[_feq_index]); 
  const fptype feq = RO_CACHE(functorConstants[cIndex]);
  const fptype kappa = g2*mu + 3*(1+v1)*feq*sigmaT*sigmaT* mu*mu;
    
  /* Rewrite a and b in terms of charge_mean and variance */ 
  const fptype b = kappa / (3*variance);
  const fptype a = variance - b*mu - b*b;

  if(a+b*mu<=0) return 0;
  const fptype sigma_inv = 1.0/SQRT(a+b*Evis); 
  fptype arg = (Evis-mu+b)*sigma_inv;
  arg *= -0.5*arg;
  fptype ret = 0.3989422804 * sigma_inv*EXP(arg);
#ifdef RPF_CHECK
  {
    const fptype Eraw = (evt[RO_CACHE(indices[RO_CACHE(indices[0])+3])]);
    const fptype ly = RO_CACHE(p[RO_CACHE(indices[_NL_index+0])]);  /* light yield, npe/keV */ 
    const fptype qc1_ = RO_CACHE(p[RO_CACHE(indices[_NL_index+1])]); 
    const fptype qc2_ = RO_CACHE(p[RO_CACHE(indices[_NL_index+2])]); 
    const int _NL_size = RO_CACHE(indices[1]);  /* light yield, npe/keV */ 
    const int _Res_index = _NL_index+_NL_size+1;  /* light yield, npe/keV */ 
    const fptype v1 = RO_CACHE(p[RO_CACHE(indices[_Res_index+0])]);  
    const fptype sigmaT = RO_CACHE(p[RO_CACHE(indices[_Res_index+1])]); 
    const int _Res_size = RO_CACHE(indices[_NL_index+_NL_size]);  /* light yield, npe/keV */ 
    const int _feq_index = _Res_index+_Res_size;  /* light yield, npe/keV */ 
    const unsigned int cIndex = RO_CACHE(indices[_feq_index]); 
    const fptype feq = RO_CACHE(functorConstants[cIndex]);
    if(fabs(mu-Evis)<sqrt(variance)*0.2) 
      printf("%d %.1lf <- %.2lf : (%.1lf %.3lf %.3lf) (%.3lf %.1lf %.1lf) (%.3lf) | (%lf %lf %lf) (%lf %lf) -> %lf\n", 
	  THREADIDX, Evis, /**/ ly, qc1_, qc2_, /**/ v1, sigmaT, g2, /**/ feq, /**/ mu, variance, kappa, /**/ b,a, /**/ ret);
  }
#endif
  return ret; 
} 
MEM_DEVICE device_function_ptr ptr_to_npe_GeneralizedGamma_Mach4_normal = device_npe_GeneralizedGamma <ResponseFunctionPdf::NL::Mach4, ResponseFunctionPdf::Mean::normal, ResponseFunctionPdf::RES::charge>;
MEM_DEVICE device_function_ptr ptr_to_npe_GeneralizedGamma_Mach4_peak = device_npe_GeneralizedGamma <ResponseFunctionPdf::NL::Mach4, ResponseFunctionPdf::Mean::peak, ResponseFunctionPdf::RES::charge>;
MEM_DEVICE device_function_ptr ptr_to_npe_GeneralizedGamma_Mach4_shifted = device_npe_GeneralizedGamma <ResponseFunctionPdf::NL::Mach4, ResponseFunctionPdf::Mean::shifted, ResponseFunctionPdf::RES::charge>;
MEM_DEVICE device_function_ptr ptr_to_npe_GeneralizedGamma_expPar_normal = device_npe_GeneralizedGamma <ResponseFunctionPdf::NL::expPar, ResponseFunctionPdf::Mean::normal, ResponseFunctionPdf::RES::charge>;
MEM_DEVICE device_function_ptr ptr_to_npe_GeneralizedGamma_expPar_peak = device_npe_GeneralizedGamma <ResponseFunctionPdf::NL::expPar, ResponseFunctionPdf::Mean::peak, ResponseFunctionPdf::RES::charge>;
MEM_DEVICE device_function_ptr ptr_to_npe_GeneralizedGamma_expPar_shifted = device_npe_GeneralizedGamma <ResponseFunctionPdf::NL::expPar, ResponseFunctionPdf::Mean::shifted, ResponseFunctionPdf::RES::charge>;
MEM_DEVICE device_function_ptr ptr_to_npe_ModifiedGaussian_Mach4_normal = device_npe_ModifiedGaussian <ResponseFunctionPdf::NL::Mach4, ResponseFunctionPdf::Mean::normal, ResponseFunctionPdf::RES::charge>;
MEM_DEVICE device_function_ptr ptr_to_npe_ModifiedGaussian_Mach4_peak = device_npe_ModifiedGaussian <ResponseFunctionPdf::NL::Mach4, ResponseFunctionPdf::Mean::peak, ResponseFunctionPdf::RES::charge>;
MEM_DEVICE device_function_ptr ptr_to_npe_ModifiedGaussian_Mach4_shifted = device_npe_ModifiedGaussian <ResponseFunctionPdf::NL::Mach4, ResponseFunctionPdf::Mean::shifted, ResponseFunctionPdf::RES::charge>;
MEM_DEVICE device_function_ptr ptr_to_npe_ModifiedGaussian_expPar_normal = device_npe_ModifiedGaussian <ResponseFunctionPdf::NL::expPar, ResponseFunctionPdf::Mean::normal, ResponseFunctionPdf::RES::charge>;
MEM_DEVICE device_function_ptr ptr_to_npe_ModifiedGaussian_expPar_peak = device_npe_ModifiedGaussian <ResponseFunctionPdf::NL::expPar, ResponseFunctionPdf::Mean::peak, ResponseFunctionPdf::RES::charge>;
MEM_DEVICE device_function_ptr ptr_to_npe_ModifiedGaussian_expPar_shifted = device_npe_ModifiedGaussian <ResponseFunctionPdf::NL::expPar, ResponseFunctionPdf::Mean::shifted, ResponseFunctionPdf::RES::charge>;
