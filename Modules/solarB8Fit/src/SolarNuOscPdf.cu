/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "SolarNuOscPdf.h"

template<SolarNuOscPdf::PeeType T>
EXEC_TARGET fptype SolarNuPee(
  const fptype E_nu,const fptype Ne,const fptype deltaM21,const fptype ,const fptype sin_2theta12,const fptype sin_2theta13);

template<SolarNuOscPdf::PeeType T>
EXEC_TARGET fptype device_SolarNuOsc(fptype* evt, fptype* p, unsigned int* indices) {
  const fptype eNeu = evt[RO_CACHE(indices[2 + RO_CACHE(indices[0])])]; 
  const fptype sinTheta12_2 = RO_CACHE(p[RO_CACHE(indices[1])]);
  const fptype sinTheta13_2 = RO_CACHE(p[RO_CACHE(indices[2])]);
  const fptype deltaM221 = RO_CACHE(p[RO_CACHE(indices[3])])*1e-5;
  const fptype deltaM231 = RO_CACHE(p[RO_CACHE(indices[4])])*1e-3;
  const fptype Ne = RO_CACHE(functorConstants[RO_CACHE(indices[5])]); // mol/cm^3
  return SolarNuPee<T>(eNeu,Ne,deltaM221,deltaM231,sinTheta12_2,sinTheta13_2);
}
template<>
EXEC_TARGET fptype SolarNuPee<SolarNuOscPdf::PeeType::Simple>(
  const fptype E_nu,const fptype Ne,const fptype deltaM21,const fptype ,const fptype sin_2theta12,const fptype sin_2theta13) {
  const fptype G_Fermi = 1.166e-5*1e-18; // in eV^{-2}
  const fptype one = 197.3e-9; // 1 = 197e-9 m*eV
  const fptype Na = 6.022e23; // mol^{-1}
  const fptype MeV = 1e6; // in eV
  const fptype cm = 1e-2; // in m

  fptype Acc = 2*sqrt(2.)*G_Fermi*pow(one,3)*Na*(E_nu*MeV)*Ne*pow(cm,-3);  //eV^2, 2sqrt(2)E*G_FN_e, N_e = 100 N_A/cm3
  fptype cos2theta12 = 1-2*sin_2theta12;
  fptype sin2theta12 = 2*sqrt(sin_2theta12)*sqrt(1-sin_2theta12);
  fptype deltaM = sqrt(pow(deltaM21*cos2theta12-Acc,2)+pow(deltaM21*sin2theta12,2));
  fptype cos2thetaM = (deltaM21*cos2theta12-Acc)/deltaM;
  fptype sin_4theta13 = pow(sin_2theta13,2);
  fptype cos_4theta13 = pow(1-sin_2theta13,2);
  return (0.5+0.5*cos2thetaM*cos2theta12)*cos_4theta13+sin_4theta13;
}
template<>
EXEC_TARGET fptype SolarNuPee<SolarNuOscPdf::PeeType::Full>(
  const fptype E,const fptype Ne,const fptype deltaM2_21,const fptype deltaM2_31,const fptype s2_12,const fptype s2_13) {
  const fptype G_Fermi = 1.166e-5*1e-18; // in eV^{-2}
  const fptype one = 197.3e-9; // 1 = 197e-9 m*eV
  const fptype Na = 6.022e23; // mol^{-1}
  const fptype MeV = 1e6; // in eV
  const fptype cm = 1e-2; // in m
  const fptype Vsun = sqrt(2.)*G_Fermi/*eV^{-2}*/*pow(one,3)/*m^3 eV^3*/*Ne*pow(cm,-3)/*mol m^-3*/*Na/*/mol*//MeV; // in MeV
  fptype ep_sun_12 = 2*E*Vsun*1e12/deltaM2_21;
  fptype ep_sun_13 = 2*E*Vsun*1e12/deltaM2_31;
  // no earth
  fptype c_212 = 1-2*s2_12;
  fptype c2_13 = 1-s2_13;
  fptype s2_212 = 1-c_212*c_212;
  fptype cos_m = sqrt(pow(c_212-c2_13*ep_sun_12,2)+s2_212);
  fptype c_212m = (c_212-c2_13*ep_sun_12)/cos_m;
  fptype s2_13m = s2_13*(1+2*ep_sun_13);
  fptype c2_13m = 1-s2_13m;

  auto Pee = [=](fptype freg) {
    return (c2_13*(1-s2_12)-freg)*c2_13m*c_212m+c2_13*c2_13m*((1-c_212m)/2)+s2_13*s2_13m;
  };
  return Pee(0);
}

MEM_DEVICE device_function_ptr ptr_to_SolarNuOsc_simple = device_SolarNuOsc<SolarNuOscPdf::PeeType::Simple>; 
MEM_DEVICE device_function_ptr ptr_to_SolarNuOsc_full = device_SolarNuOsc<SolarNuOscPdf::PeeType::Full>; 

  __host__ SolarNuOscPdf::SolarNuOscPdf (std::string n, Variable *eNeu, std::vector<Variable*> sinTheta_2s, std::vector<Variable*>deltaM2s, fptype Ne,bool useFull)
: GooPdf(eNeu, n) 
{
  std::vector<unsigned int> pindices;
  for(auto sinTheata_2 : sinTheta_2s) {
    pindices.push_back(registerParameter(sinTheata_2));
  }
  for(auto deltaM2 : deltaM2s) {
    pindices.push_back(registerParameter(deltaM2));
  }
  pindices.push_back(registerConstants(1));
  MEMCPY_TO_SYMBOL(functorConstants, &Ne, sizeof(fptype), cIndex*sizeof(fptype), cudaMemcpyHostToDevice); 
  if(useFull)
    GET_FUNCTION_ADDR(ptr_to_SolarNuOsc_full);
  else
    GET_FUNCTION_ADDR(ptr_to_SolarNuOsc_simple);
  initialise(pindices); 
}


__host__ fptype SolarNuOscPdf::normalise () const {
  host_normalisation[parameters] = 1.0;
  return 1; 
}
