/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "Mach4GGResponseFunctionPdf.h"

// Nonlinearity model reference: (Mach4 Quenching)
//   [1] R. N. Saldanha, “Precision Measurement of the 7 Be Solar Neutrino Interaction Rate in Borexino,” Princeton University, 2012.
// Resolution model Reference: (Generalized Gamma function)
//   [2] O. J. Smirnov, “An approximation of the ideal scintillation detector line shape with a generalized gamma distribution,” Nucl. Instruments Methods Phys. Res. Sect. A Accel. Spectrometers, Detect. Assoc. Equip., vol. 595, no. 2, pp. 410–418, 2008.
#define DEFINE_RPF_DEVICE(NAME, GETMEAN) \
  EXEC_TARGET fptype device_npe_GeneralizedGamma_Mach4_ ## NAME (fptype* evt, fptype* p, unsigned int* indices) {  \
    /* Observables */\
    const fptype Evis = (evt[RO_CACHE(indices[RO_CACHE(indices[0])+2])]);  \
    const fptype Eraw = (evt[RO_CACHE(indices[RO_CACHE(indices[0])+3])]); \
    /* parameters follows the exact order of "registerParameter" */ \
    /* Non-linearity parameters */ \
    const fptype ly = RO_CACHE(p[RO_CACHE(indices[Mach4GG_NL_index+0])]);  /* light yield, npe/keV */ \
    const fptype qc1_ = RO_CACHE(p[RO_CACHE(indices[Mach4GG_NL_index+1])]); \
    const fptype qc2_ = RO_CACHE(p[RO_CACHE(indices[Mach4GG_NL_index+2])]); \
    /* Resolution parameters */ \
    const fptype v1 = RO_CACHE(p[RO_CACHE(indices[Mach4GG_Res_index+0])]);  \
    const fptype vT = RO_CACHE(p[RO_CACHE(indices[Mach4GG_Res_index+1])]); \
    /* feq : equalization factor introduced to remove loss of channels */ \
    const unsigned int cIndex = RO_CACHE(indices[Mach4GG_feq_index]); \
    const fptype feq = RO_CACHE(functorConstants[cIndex]);\
    \
    /* calculate mu */ \
    fptype mu; \
    { GETMEAN; }; \
    /* calculate var */ \
    if(mu<=0.5) return -1e10; \
    const fptype variance = mu*(1.+v1) *feq + vT*1e-4*mu*mu;\
    \
    /* Rewrite moments of ideal detector response in terms of charge_mean and variance */ \
    const fptype moment_2 = mu*mu + variance; \
    const fptype moment_4 = (variance*variance*(2.+3./mu) + 4.*mu*mu*(variance - 2.) \
	+ 2.*mu*(6.*variance - 1) \
	+ (variance + mu*mu)*(variance + mu*mu)); \
    \
    /* get alpha and beta from momentums */ \
    const fptype alpha = moment_2*moment_2/(moment_4 - moment_2*moment_2); \
    const fptype beta = alpha/moment_2; \
/*    printf("%d %.1lf <- %.2lf : (%.1lf %.3lf %.3lf) (%.3lf %.1lf) (%.3lf) | (%lf %lf) (%lf %lf) (%lf %lf)\n", */\
/*	   THREADIDX, Evis, Eraw, ly, qc1_, qc2_, v1, vT, feq, mu, variance, moment_2, moment_4, alpha, beta ); */\
    return (EXP(LOG(2.) \
	  + alpha*LOG(beta) \
	  + LOG(Evis)*(2.*alpha -1.) \
	  + -beta*Evis*Evis \
	  - ::lgamma(alpha)));  \
  } \
MEM_DEVICE device_function_ptr ptr_to_npe_GeneralizedGamma_Mach4_ ## NAME \
= device_npe_GeneralizedGamma_Mach4_ ## NAME;

#define MACH4 \
{ mu = (1-POW(1+Eraw*1000/qc1_,qc2_))*ly*Eraw; /* Eraw in MeV */ }
#define MACH4_SHIFTED \
{ MACH4; \
  mu += RO_CACHE(p[RO_CACHE(indices[Mach4GG_shiftE_index+0])]);  /* light yield */ } 
#define MACH4_PEAK \
{ mu = RO_CACHE(p[RO_CACHE(indices[Mach4GG_peakE_index+0])]);  /* light yield */ } \

DEFINE_RPF_DEVICE(normal, MACH4 );
DEFINE_RPF_DEVICE(shifted, MACH4_SHIFTED);
DEFINE_RPF_DEVICE(peak, MACH4_PEAK);
