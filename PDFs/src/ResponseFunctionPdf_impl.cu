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

// Nonlinearity model and resolution model are in ResponseFunctionPdf_NLRES.cu
//   O. J. Smirnov, “An approximation of the ideal scintillation detector line shape with a generalized gamma distribution,” Nucl. Instruments Methods Phys. Res. Sect. A Accel. Spectrometers, Detect. Assoc. Equip., vol. 595, no. 2, pp. 410–418, 2008.
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
  if(fabs(mu-Evis)<sqrt(variance)*0.2) 
    printf("%d %.1lf <- %.2lf : (%.1lf %.3lf %.3lf) (%.3lf %.1lf) (%.3lf) | (%lf %lf) (%lf %lf) (%lf %lf) -> %lf\n", 
	   THREADIDX, Evis, Eraw, ly, qc1_, qc2_, v1, vT, feq, mu, variance, moment_2, moment_4, alpha, beta, ret ); 
#endif
  return ret; 
} 
MEM_DEVICE device_function_ptr ptr_to_npe_GeneralizedGamma_Mach4_normal = device_npe_GeneralizedGamma <ResponseFunctionPdf::NL::Mach4, ResponseFunctionPdf::Mean::normal, ResponseFunctionPdf::RES::charge>;
MEM_DEVICE device_function_ptr ptr_to_npe_GeneralizedGamma_Mach4_peak = device_npe_GeneralizedGamma <ResponseFunctionPdf::NL::Mach4, ResponseFunctionPdf::Mean::peak, ResponseFunctionPdf::RES::charge>;
MEM_DEVICE device_function_ptr ptr_to_npe_GeneralizedGamma_Mach4_shifted = device_npe_GeneralizedGamma <ResponseFunctionPdf::NL::Mach4, ResponseFunctionPdf::Mean::shifted, ResponseFunctionPdf::RES::charge>;
