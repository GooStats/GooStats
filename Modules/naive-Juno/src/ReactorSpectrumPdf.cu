/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "ReactorSpectrumPdf.h"

EXEC_TARGET fptype device_ReactorSpectrum (fptype* evt, fptype* p, unsigned int* indices) {
  const fptype E = evt[RO_CACHE(indices[2 + RO_CACHE(indices[0])])]; // in MeV
  const fptype U235 =  RO_CACHE(p[RO_CACHE(indices[1])]);
  const fptype U238 =  RO_CACHE(p[RO_CACHE(indices[2])]);
  const fptype Pu239 = RO_CACHE(p[RO_CACHE(indices[3])]);
  //const fptype Pu241 = RO_CACHE(p[RO_CACHE(indices[4])]);
  const fptype Pu241 = 1-U235-U238-Pu239;
  const unsigned int cIndex = RO_CACHE(indices[5]);
  const fptype *U235p = functorConstants+cIndex;
  const fptype *U238p = functorConstants+cIndex+3;
  const fptype *Pu239p = functorConstants+cIndex+6;
  const fptype *Pu241p = functorConstants+cIndex+9;
  const fptype phiU235 = EXP(RO_CACHE(U235p[0]) + RO_CACHE(U235p[1])*E + RO_CACHE(U235p[2])*E*E);
  const fptype phiU238 = EXP(RO_CACHE(U238p[0]) + RO_CACHE(U238p[1])*E + RO_CACHE(U238p[2])*E*E);
  const fptype phiPu239 = EXP(RO_CACHE(Pu239p[0]) + RO_CACHE(Pu239p[1])*E + RO_CACHE(Pu239p[2])*E*E);
  const fptype phiPu241 = EXP(RO_CACHE(Pu241p[0]) + RO_CACHE(Pu241p[1])*E + RO_CACHE(Pu241p[2])*E*E);
  const fptype ret = phiU235*U235+phiU238*U238+phiPu239*Pu239+phiPu241*Pu241;
//#define DEBUG
#ifdef DEBUG
  printf("%d %lf -> (%lf / %lf %lf %lf) %lf %lf %lf %lf %lf\n",THREADIDX, E, 
	 U235,U235p[0],U235p[1],U235p[2],
	 phiU235,phiU238,Pu239,Pu241,ret);
#endif
  return ret;
}

MEM_DEVICE device_function_ptr ptr_to_ReactorSpectrum = device_ReactorSpectrum; 

__host__ ReactorSpectrumPdf::ReactorSpectrumPdf (std::string n, Variable *_x,
    const std::vector<Variable*> &fractions,const std::vector<double> &coefficients)
  : GooPdf(_x, n) 
{
  std::vector<unsigned int> pindices;
  for(auto fraction : fractions)
    pindices.push_back(registerParameter(fraction));
  GET_FUNCTION_ADDR(ptr_to_ReactorSpectrum);
  pindices.push_back(registerConstants(12));/*6*/ 
  fptype toCopy[12];
  for(int i = 0;i<12;++i) toCopy[i] = coefficients[i];
  MEMCPY_TO_SYMBOL(functorConstants, toCopy, 12*sizeof(fptype), cIndex*sizeof(fptype), cudaMemcpyHostToDevice); 
  initialise(pindices); 
}
__host__ fptype ReactorSpectrumPdf::normalise () const {
  host_normalisation[parameters] = 1.0;
  return 1; 
}
