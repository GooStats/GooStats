/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "RangedExpPdf.h"
#include "goofit/Variable.h"

EXEC_TARGET fptype device_RangedExpPdfs (fptype* evt, fptype* p, unsigned int* indices) { 
  const fptype x = evt[RO_CACHE(indices[2 + RO_CACHE(indices[0])])]; 
  const fptype p0 = RO_CACHE(p[RO_CACHE(indices[1])]); 
  const fptype p1 = RO_CACHE(p[RO_CACHE(indices[2])]); 
  const int cIndex = RO_CACHE(indices[3]); 
  const fptype x_L = RO_CACHE(functorConstants[cIndex]);
  const fptype x_H = RO_CACHE(functorConstants[cIndex+1]);

  fptype ret = (x>=x_L && x<=x_H)?p0*EXP(-x/p1):0; 
  //if(THREADIDX==0) printf("p0: %lf %lf %lf %lf %lf %lf\n",x,p0,p1,x_L,x_H,ret);
  return ret; 
}
EXEC_TARGET fptype device_inverseRangedExpPdfs (fptype* evt, fptype* p, unsigned int* indices) { 
  const fptype x = evt[RO_CACHE(indices[2 + RO_CACHE(indices[0])])]; 
  const fptype p0 = RO_CACHE(p[RO_CACHE(indices[1])]); 
  const fptype p1 = RO_CACHE(p[RO_CACHE(indices[2])]); 
  const int cIndex = RO_CACHE(indices[3]); 
  const fptype x_L = RO_CACHE(functorConstants[cIndex]);
  const fptype x_H = RO_CACHE(functorConstants[cIndex+1]);

  fptype ret = (x>=x_L && x<=x_H)?1-p0*EXP(-x/p1):1; 
  //if(THREADIDX==0) printf("1-p0: %lf %lf %lf %lf %lf %lf\n",x,p0,p1,x_L,x_H,ret);
  return ret; 
}

MEM_DEVICE device_function_ptr ptr_to_RangedExpPdfs = device_RangedExpPdfs; 
MEM_DEVICE device_function_ptr ptr_to_inverseRangedExpPdfs = device_inverseRangedExpPdfs; 

RangedExpPdf::RangedExpPdf (std::string n, Variable *npe,Variable* p0, Variable* p1, fptype x_L, fptype x_H, bool inversed) 
  : GooPdf(npe, n) 
{
  std::cout<<"RangedExpPdf("<<n<<"): p0 "<<p0->value<<" p1 "<<p1->value<<" xL "<<x_L<<" xH "<<x_H<<std::endl;
  pindices.push_back(registerParameter(p0));
  pindices.push_back(registerParameter(p1));
  pindices.push_back(registerConstants(2));
  fptype constants[2] = { x_L, x_H };
  MEMCPY_TO_SYMBOL(functorConstants, constants, sizeof(constants), cIndex*sizeof(fptype), cudaMemcpyHostToDevice); 
  if(!inversed) { GET_FUNCTION_ADDR(ptr_to_RangedExpPdfs); } else { GET_FUNCTION_ADDR(ptr_to_inverseRangedExpPdfs); }
  initialise(pindices); 
}
__host__ fptype RangedExpPdf::normalise () const {
  host_normalisation[parameters] = 1.0; 
  return 1; 
}

