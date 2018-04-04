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

__host__ Mach4GGResponseFunctionPdf::Mach4GGResponseFunctionPdf (std::string n, 
	Variable* npe, Variable *energy, 
        string response_function, string quenching_model, 
	std::vector<Variable*> NL,
	std::vector<Variable*> res,
	double feq) : GooPdf(npe,n) {
  registerObservable(energy);
  insertResponseFunctionAndNLPar(NL,res,feq);
  chooseFunctionPtr(npe,response_function,quenching_model,RPFtype::normal); 
  initialise(pindices);  
}
__host__ Mach4GGResponseFunctionPdf::Mach4GGResponseFunctionPdf (std::string n, 
	Variable* npe, Variable *energy, 
        string response_function, string quenching_model, 
	std::vector<Variable*> NL,
	std::vector<Variable*> res,
	double feq,
	Variable *npeShift) : GooPdf(npe,n) {
  registerObservable(energy);
  insertResponseFunctionAndNLPar(NL,res,feq);
  pindices.push_back(registerParameter(npeShift)); 
  chooseFunctionPtr(npe,response_function,quenching_model,RPFtype::shifted); 
  initialise(pindices);  
}
__host__ Mach4GGResponseFunctionPdf::Mach4GGResponseFunctionPdf (std::string n, 
	Variable* npe, Variable *energy, 
        string response_function, string quenching_model, 
	std::vector<Variable*> res,
	double feq,
	Variable *peakEvis) : GooPdf(npe,n) {
  registerObservable(energy);
  std::vector<Variable*> dummy; for(int i = 0;i<3;++i) dummy.push_back(nullptr);
  insertResponseFunctionAndNLPar(dummy,res,feq);
  pindices.push_back(registerParameter(peakEvis));
  chooseFunctionPtr(npe,response_function,quenching_model,RPFtype::peak); 
  initialise(pindices);  
}
void Mach4GGResponseFunctionPdf::insertResponseFunctionAndNLPar(
    const std::vector<Variable*> &NL,const std::vector<Variable*> &res,double feq) {
  std::cout<<"Dumping NL of <"<<getName()<<">:"<<std::endl;
  if(NL.size()!= ((Mach4GG_Res_index)-(Mach4GG_NL_index))) {
    abortWithCudaPrintFlush(__FILE__, __LINE__, getName() + 
    "size of NL is inconsistent with (macro) Mach4GG_Res_index - Mach4GG_NL_index!");
  }
  for(unsigned int i = 0;i<NL.size();++i) { 
    std::cout<<NL.at(i)->name<<" : "<<NL.at(i)<<std::endl;
    pindices.push_back(NL.at(i)?registerParameter(NL.at(i)):0);/*1--3*/
  }
  if(res.size()!= ((Mach4GG_feq_index)-(Mach4GG_Res_index))) {
    abortWithCudaPrintFlush(__FILE__, __LINE__, getName() + 
    "size of res is inconsistent with (macro) Mach4GG_feq_index - Mach4GG_Res_index!");
  }
  std::cout<<"Dumping res of <"<<getName()<<">:"<<std::endl;
  for(unsigned int i = 0;i<res.size();++i) { 
    std::cout<<res.at(i)->name<<" : "<<res.at(i)<<std::endl;
    pindices.push_back(registerParameter(res.at(i)));/*4--5*/ 
  } 
  if(1 != ((Mach4GG_shiftE_index)-(Mach4GG_feq_index)) || 
      1 != ((Mach4GG_peakE_index)-(Mach4GG_feq_index)) ) {
    abortWithCudaPrintFlush(__FILE__, __LINE__, getName() + 
    "1 is inconsistent with (macro) Mach4GG_shiftE_index/Mach4GG_peakE_index - Mach4GG_Res_index!");
  }
  pindices.push_back(registerConstants(1));/*6*/ 
  MEMCPY_TO_SYMBOL(functorConstants, &feq, sizeof(fptype), cIndex*sizeof(fptype), cudaMemcpyHostToDevice); 
}
extern MEM_DEVICE device_function_ptr ptr_to_npe_GeneralizedGamma_Mach4_normal;
extern MEM_DEVICE device_function_ptr ptr_to_npe_GeneralizedGamma_Mach4_shifted;
extern MEM_DEVICE device_function_ptr ptr_to_npe_GeneralizedGamma_Mach4_peak;
void Mach4GGResponseFunctionPdf::chooseFunctionPtr(Variable *,const std::string &response_function,const std::string &quenching_model,const RPFtype rpf_type) const {
  if(!(quenching_model == "Mach4" && response_function == "GG")) 
    abortWithCudaPrintFlush(__FILE__, __LINE__, getName() + " Only Mach4 + Generalized Gamma are implemented in this class. For more response function, please wait.");
  switch(rpf_type) {
    case RPFtype::normal:
      GET_FUNCTION_ADDR(ptr_to_npe_GeneralizedGamma_Mach4_normal);
      break;
    case RPFtype::shifted:
      GET_FUNCTION_ADDR(ptr_to_npe_GeneralizedGamma_Mach4_shifted);
      break;
    case RPFtype::peak:
      GET_FUNCTION_ADDR(ptr_to_npe_GeneralizedGamma_Mach4_peak);
      break;
    default:
      abortWithCudaPrintFlush(__FILE__, __LINE__, getName() + " unknown RPF type. checking your code", this);  
  }
}
