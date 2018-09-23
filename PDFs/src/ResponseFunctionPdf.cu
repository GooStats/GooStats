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

__host__ ResponseFunctionPdf::ResponseFunctionPdf (std::string n, 
	Variable* npe, Variable *energy, 
        string response_function, string quenching_model, 
	std::vector<Variable*> NL,
	std::vector<Variable*> res,
	double feq) : GooPdf(npe,n) {
  registerObservable(energy);
  insertResponseFunctionAndNLPar(NL,res,feq);
  chooseFunctionPtr(npe,response_function,quenching_model,Mean::normal); 
  initialise(pindices);  
}
__host__ ResponseFunctionPdf::ResponseFunctionPdf (std::string n, 
	Variable* npe, Variable *energy, 
        string response_function, string quenching_model, 
	std::vector<Variable*> NL,
	std::vector<Variable*> res,
	double feq,
	Variable *npeShift) : GooPdf(npe,n) {
  registerObservable(energy);
  insertResponseFunctionAndNLPar(NL,res,feq);
  pindices.push_back(registerParameter(npeShift)); 
  chooseFunctionPtr(npe,response_function,quenching_model,Mean::shifted); 
  initialise(pindices);  
}
__host__ ResponseFunctionPdf::ResponseFunctionPdf (std::string n, 
	Variable* npe, Variable *energy, 
        string response_function, string quenching_model, 
	std::vector<Variable*> res,
	double feq,
	Variable *peakEvis) : GooPdf(npe,n) {
  registerObservable(energy);
  std::vector<Variable*> dummy; 
  insertResponseFunctionAndNLPar(dummy,res,feq);
  pindices.push_back(registerParameter(peakEvis));
  chooseFunctionPtr(npe,response_function,quenching_model,Mean::peak); 
  initialise(pindices);  
}
void ResponseFunctionPdf::insertResponseFunctionAndNLPar(
    const std::vector<Variable*> &NL,const std::vector<Variable*> &res,double feq) {
  std::cout<<"Dumping NL of <"<<getName()<<">:"<<std::endl;
  pindices.push_back(NL.size());
  for(unsigned int i = 0;i<NL.size();++i) { 
    std::cout<<NL.at(i)->name<<" : "<<NL.at(i)->value<<" ± "<<NL.at(i)->error<<" ( "<<NL.at(i)->lowerlimit<<" , "<<NL.at(i)->upperlimit<<" )"<<std::endl;
    pindices.push_back(registerParameter(NL.at(i)));/*1--3*/
  }
  pindices.push_back(res.size());
  std::cout<<"Dumping res of <"<<getName()<<">:"<<std::endl;
  for(unsigned int i = 0;i<res.size();++i) { 
    std::cout<<res.at(i)->name<<" : "<<res.at(i)->value<<" ± "<<res.at(i)->error<<" ( "<<res.at(i)->lowerlimit<<" , "<<res.at(i)->upperlimit<<" )"<<std::endl;
    pindices.push_back(registerParameter(res.at(i)));/*4--5*/ 
  } 
  pindices.push_back(registerConstants(1));/*6*/ 
  std::cout<<"Dumping feq of <"<<getName()<<">: "<<feq<<std::endl;
  MEMCPY_TO_SYMBOL(functorConstants, &feq, sizeof(fptype), cIndex*sizeof(fptype), cudaMemcpyHostToDevice); 
}
extern MEM_DEVICE device_function_ptr ptr_to_npe_GeneralizedGamma_Mach4_normal;
extern MEM_DEVICE device_function_ptr ptr_to_npe_GeneralizedGamma_Mach4_peak;
extern MEM_DEVICE device_function_ptr ptr_to_npe_GeneralizedGamma_Mach4_shifted;
extern MEM_DEVICE device_function_ptr ptr_to_npe_GeneralizedGamma_Echidna_normal;
extern MEM_DEVICE device_function_ptr ptr_to_npe_GeneralizedGamma_Echidna_peak;
extern MEM_DEVICE device_function_ptr ptr_to_npe_GeneralizedGamma_Echidna_shifted;
extern MEM_DEVICE device_function_ptr ptr_to_npe_GeneralizedGamma_expPar_normal;
extern MEM_DEVICE device_function_ptr ptr_to_npe_GeneralizedGamma_expPar_peak;
extern MEM_DEVICE device_function_ptr ptr_to_npe_GeneralizedGamma_expPar_shifted;
extern MEM_DEVICE device_function_ptr ptr_to_npe_ModifiedGaussian_Mach4_normal;
extern MEM_DEVICE device_function_ptr ptr_to_npe_ModifiedGaussian_Mach4_peak;
extern MEM_DEVICE device_function_ptr ptr_to_npe_ModifiedGaussian_Mach4_shifted;
extern MEM_DEVICE device_function_ptr ptr_to_npe_ModifiedGaussian_Echidna_normal;
extern MEM_DEVICE device_function_ptr ptr_to_npe_ModifiedGaussian_Echidna_peak;
extern MEM_DEVICE device_function_ptr ptr_to_npe_ModifiedGaussian_Echidna_shifted;
extern MEM_DEVICE device_function_ptr ptr_to_npe_ModifiedGaussian_expPar_normal;
extern MEM_DEVICE device_function_ptr ptr_to_npe_ModifiedGaussian_expPar_peak;
extern MEM_DEVICE device_function_ptr ptr_to_npe_ModifiedGaussian_expPar_shifted;
extern MEM_DEVICE device_function_ptr ptr_to_npe_ScaledPoisson_Mach4_normal;
extern MEM_DEVICE device_function_ptr ptr_to_npe_ScaledPoisson_Mach4_peak;
extern MEM_DEVICE device_function_ptr ptr_to_npe_ScaledPoisson_Mach4_shifted;
extern MEM_DEVICE device_function_ptr ptr_to_npe_ScaledPoisson_Echidna_normal;
extern MEM_DEVICE device_function_ptr ptr_to_npe_ScaledPoisson_Echidna_peak;
extern MEM_DEVICE device_function_ptr ptr_to_npe_ScaledPoisson_Echidna_shifted;
extern MEM_DEVICE device_function_ptr ptr_to_npe_ScaledPoisson_expPar_normal;
extern MEM_DEVICE device_function_ptr ptr_to_npe_ScaledPoisson_expPar_peak;
extern MEM_DEVICE device_function_ptr ptr_to_npe_ScaledPoisson_expPar_shifted;
void ResponseFunctionPdf::chooseFunctionPtr(Variable *,const std::string &response_function,const std::string &quenching_model,const Mean rpf_type) const {
std::cout<<"Choosing RPF<"<<response_function<<"> NL<"<<quenching_model<<"> type<"<<static_cast<int>(rpf_type)<<"> for ["<<getName()<<"]"<<std::endl;
  if(!(quenching_model == "Mach4" || quenching_model == "Echidna" || quenching_model == "expPar"))
    abortWithCudaPrintFlush(__FILE__, __LINE__, getName() + " Only Mach4/Echidna/expPar NL model are implemented in this class. If not enough, please post an github issue.");
  if(!(response_function == "GeneralizedGamma" || response_function == "ModifiedGaussian" || response_function == "ScaledPoisson"))
    abortWithCudaPrintFlush(__FILE__, __LINE__, getName() + " Only GeneralizedGamma/ModifiedGaussian/ScaledPoisson RPF model are implemented in this class. If not enough, please post an github issue.");
#define CHOOSE_RPF(RPF,TYPE,NL) if((response_function == #RPF)&&(quenching_model == #NL)&&(rpf_type==Mean::TYPE)) GET_FUNCTION_ADDR(ptr_to_npe_#RPF#_#NL#_#TYPE);
  CHOOSE_RPF(GeneralizedGamma,Mach4,normal);
  CHOOSE_RPF(GeneralizedGamma,Mach4,shifted);
  CHOOSE_RPF(GeneralizedGamma,Mach4,peak);
  CHOOSE_RPF(GeneralizedGamma,Echidna,normal);
  CHOOSE_RPF(GeneralizedGamma,Echidna,shifted);
  CHOOSE_RPF(GeneralizedGamma,Echidna,peak);
  CHOOSE_RPF(GeneralizedGamma,expPar,normal);
  CHOOSE_RPF(GeneralizedGamma,expPar,normal);
  CHOOSE_RPF(GeneralizedGamma,expPar,normal);
  CHOOSE_RPF(ModifiedGaussian,Mach4,normal);
  CHOOSE_RPF(ModifiedGaussian,Mach4,shifted);
  CHOOSE_RPF(ModifiedGaussian,Mach4,peak);
  CHOOSE_RPF(ModifiedGaussian,Echidna,normal);
  CHOOSE_RPF(ModifiedGaussian,Echidna,shifted);
  CHOOSE_RPF(ModifiedGaussian,Echidna,peak);
  CHOOSE_RPF(ModifiedGaussian,expPar,normal);
  CHOOSE_RPF(ModifiedGaussian,expPar,normal);
  CHOOSE_RPF(ModifiedGaussian,expPar,normal);
  CHOOSE_RPF(ScaledPoisson,Mach4,normal);
  CHOOSE_RPF(ScaledPoisson,Mach4,shifted);
  CHOOSE_RPF(ScaledPoisson,Mach4,peak);
  CHOOSE_RPF(ScaledPoisson,Echidna,normal);
  CHOOSE_RPF(ScaledPoisson,Echidna,shifted);
  CHOOSE_RPF(ScaledPoisson,Echidna,peak);
  CHOOSE_RPF(ScaledPoisson,expPar,normal);
  CHOOSE_RPF(ScaledPoisson,expPar,normal);
  CHOOSE_RPF(ScaledPoisson,expPar,normal);
}
