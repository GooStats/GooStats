/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef ResponseFunctionPdf_H
#define ResponseFunctionPdf_H

#include "goofit/PDFs/GooPdf.h" 
// 0 correspond to the index of observables
#define _NL_index 2

class ResponseFunctionPdf : public GooPdf {
  public:
    // normal
    ResponseFunctionPdf (std::string n, 
	Variable* npe, Variable *energy, 
        string response_function, string quenching_model, 
	std::vector<Variable*> NL,
	std::vector<Variable*> res,
	double feq);
    // shifted
    ResponseFunctionPdf (std::string n, 
	Variable* npe, Variable *energy, 
        string response_function, string quenching_model, 
	std::vector<Variable*> NL,
	std::vector<Variable*> res,
	double feq,
	Variable *npeShift);
    // peak
    ResponseFunctionPdf (std::string n, 
	Variable* npe, Variable *energy, 
        string response_function, string quenching_model, 
	std::vector<Variable*> res,
	double feq,
	Variable *peakEvis);
    __host__ fptype integrate (fptype , fptype ) const { return 1; }
    __host__ virtual bool hasAnalyticIntegral () const {return true;} 
    enum class NL { Mach4, expPar };
    enum class Mean { normal, peak, shifted };
    enum class RES { charge, pol3 };
  private:
    void insertResponseFunctionAndNLPar(const std::vector<Variable*> &quenching_par,
	const std::vector<Variable*> &res,double feq);
    void chooseFunctionPtr(Variable *npe,const std::string &response_function,const std::string &quenching_model,const Mean rpf_type) const;
    std::vector<unsigned int> pindices; 
};

template<ResponseFunctionPdf::NL nl> EXEC_TARGET fptype GetNL(fptype *evt,fptype *p,unsigned int *indices);
template<ResponseFunctionPdf::Mean type> EXEC_TARGET fptype GetMean(fptype mu,fptype *p,unsigned int *indices);
template<ResponseFunctionPdf::RES res> EXEC_TARGET fptype GetVariance(fptype mu,fptype *p,unsigned int *indices);

#endif
