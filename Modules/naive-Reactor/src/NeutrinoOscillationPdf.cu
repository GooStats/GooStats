/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "NeutrinoOscillationPdf.h"

EXEC_TARGET fptype device_NeutrinoOscillation (fptype* evt, fptype* p, unsigned int* indices) {
  const fptype eNeu = evt[RO_CACHE(indices[2 + RO_CACHE(indices[0])])]; 
  const fptype sinTheta12_2 = RO_CACHE(p[RO_CACHE(indices[1])]);
  const fptype sinTheta13_2 = RO_CACHE(p[RO_CACHE(indices[2])]);
  //const fptype sinTheta23_2 = RO_CACHE(p[RO_CACHE(indices[3])]);
  const fptype deltaM221 = RO_CACHE(p[RO_CACHE(indices[4])])*1e-5;
  const fptype deltaM231 = RO_CACHE(p[RO_CACHE(indices[5])])*1e-3;
  const fptype deltaM232 = deltaM231-deltaM221;
  const fptype L = RO_CACHE(functorConstants[RO_CACHE(indices[6])]);

  const fptype cosTheta12_2 = 1-sinTheta12_2;
  const fptype sin2Theta12_2 = 4*sinTheta12_2*cosTheta12_2;

  const fptype cosTheta13_2 = 1-sinTheta13_2;
  const fptype cosTheta13_4 = cosTheta13_2*cosTheta13_2;
  const fptype sin2Theta13_2 = 4*sinTheta13_2*cosTheta13_2;

  const fptype Delta12 = 1.27e3*deltaM221*L/eNeu; // L in km, eNeu in MeV, deltaM2 in eV^2
  const fptype Delta13 = 1.27e3*deltaM231*L/eNeu;
  const fptype Delta23 = 1.27e3*deltaM232*L/eNeu;

  fptype ret = 1-cosTheta13_4*sin2Theta12_2*SIN(Delta12)*SIN(Delta12)
    -cosTheta12_2*sin2Theta13_2*SIN(Delta13)*SIN(Delta13)
    -sinTheta12_2*sin2Theta13_2*SIN(Delta23)*SIN(Delta23);

#ifdef RPF_CHECK
  printf("E %lf sinTheta12_2 %lf deltaM221 %lf delta12 %lf first %lf ret %lf\n",
      eNeu,sinTheta12_2, deltaM221,Delta12,cosTheta13_4*sin2Theta12_2*SIN(Delta12)*SIN(Delta12),ret);
#endif
  return ret; 
}

MEM_DEVICE device_function_ptr ptr_to_NeutrinoOscillation = device_NeutrinoOscillation; 

__host__ NeutrinoOscillationPdf::NeutrinoOscillationPdf (std::string n, Variable *eNeu, std::vector<Variable*> sinTheta_2s, std::vector<Variable*>deltaM2s, fptype distance)
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
  MEMCPY_TO_SYMBOL(functorConstants, &distance, sizeof(fptype), cIndex*sizeof(fptype), cudaMemcpyHostToDevice); 
  GET_FUNCTION_ADDR(ptr_to_NeutrinoOscillation);
  initialise(pindices); 
}


__host__ fptype NeutrinoOscillationPdf::normalise () const {
  host_normalisation[parameters] = 1.0;
  return 1; 
}
