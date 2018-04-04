/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "NewExpPdf.hh"

EXEC_TARGET fptype device_NewExp (fptype* evt, fptype* p, unsigned int* indices) {
  fptype x = evt[indices[2 + indices[0]]]; 
  fptype alpha = p[indices[1]];

  fptype ret = EXP(alpha*x*x); 
//  if(THREADIDX==0)
//    printf("newExp %d %lf %lf %lf %lf npe %.1lf alpha %lf ret %lf\n",
//	indices[1],p[indices[1]],p[0],p[1],p[2],
//	x,alpha,ret);
  return ret; 
}

MEM_DEVICE device_function_ptr ptr_to_NewExp = device_NewExp; 

__host__ NewExpPdf::NewExpPdf (std::string n, Variable* _x, Variable* alpha, Variable* ) 
  : GooPdf(_x, n) 
{
  std::vector<unsigned int> pindices;
  pindices.push_back(registerParameter(alpha));
  GET_FUNCTION_ADDR(ptr_to_NewExp);
  initialise(pindices); 
}


