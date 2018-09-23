/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "IBDPdf.h"

//Alessandro Strumia, Francesco Vissani,
//	   Precise quasielastic neutrino/nucleon cross-section,
//	   Physics Letters B,
//	   Volume 564, Issues 1–2,
//	   2003,
//	   Pages 42-54,
//	   ISSN 0370-2693,
//	   https://doi.org/10.1016/S0370-2693(03)00616-6.
//	   (http://www.sciencedirect.com/science/article/pii/S0370269303006166)

EXEC_TARGET fptype device_IBD (fptype* evt, fptype* p, unsigned int* indices) {
  const fptype eNeu = evt[RO_CACHE(indices[2 + RO_CACHE(indices[0])])]; 
  const fptype ePos = eNeu - 1.293;
  const fptype eMass = 0.511;
  const fptype pPos = (ePos>eMass)?SQRT(ePos*ePos-eMass*eMass):0;
  const fptype ret = (ePos>0)?(1e-43*pPos*ePos*POW(eNeu,-0.07056+0.02018*LOG(eNeu)-0.001953*POW(LOG(eNeu),3.0))):0; // in cm^2
#ifdef RPF_CHECK
  if((ePos>eMass)&&(THREADIDX==0))
    printf("%d eNeu %lf ePos %lf ret %le\n",THREADIDX,eNeu,ePos,ret);
#endif

  return ret; 
}

MEM_DEVICE device_function_ptr ptr_to_IBD = device_IBD; 

__host__ IBDPdf::IBDPdf (std::string n, Variable *eNeu)
  : GooPdf(eNeu, n) 
{
  std::vector<unsigned int> pindices;
  GET_FUNCTION_ADDR(ptr_to_IBD);
  initialise(pindices); 
}
__host__ fptype IBDPdf::normalise () const {
  host_normalisation[parameters] = 1.0;
  return 1; 
}
