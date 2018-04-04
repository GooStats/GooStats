/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef Mach4GGResponseFunctionPdf_H
#define Mach4GGResponseFunctionPdf_H

#include "goofit/PDFs/GooPdf.h" 
// 0 correspond to the index of observables
#define Mach4GG_NL_index 1
#define Mach4GG_Res_index 4
#define Mach4GG_feq_index 6
#define Mach4GG_shiftE_index 7
#define Mach4GG_peakE_index 7

class Mach4GGResponseFunctionPdf : public GooPdf {
  public:
    // normal
    Mach4GGResponseFunctionPdf (std::string n, 
	Variable* npe, Variable *energy, 
        string response_function, string quenching_model, 
	std::vector<Variable*> NL,
	std::vector<Variable*> res,
	double feq);
    // shifted
    Mach4GGResponseFunctionPdf (std::string n, 
	Variable* npe, Variable *energy, 
        string response_function, string quenching_model, 
	std::vector<Variable*> NL,
	std::vector<Variable*> res,
	double feq,
	Variable *npeShift);
    // peak
    Mach4GGResponseFunctionPdf (std::string n, 
	Variable* npe, Variable *energy, 
        string response_function, string quenching_model, 
	std::vector<Variable*> res,
	double feq,
	Variable *peakEvis);
    __host__ fptype integrate (fptype , fptype ) const { return 1; }
    __host__ virtual bool hasAnalyticIntegral () const {return true;} 
  private:
    void insertResponseFunctionAndNLPar(const std::vector<Variable*> &quenching_par,
	const std::vector<Variable*> &res,double feq);
    enum class RPFtype { normal, shifted, scaled, peak };
    void chooseFunctionPtr(Variable *npe,const std::string &response_function,const std::string &quenching_model,const RPFtype rpf_type) const;
    std::vector<unsigned int> pindices; 
};

#endif
