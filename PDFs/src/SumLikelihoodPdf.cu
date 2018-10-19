/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "SumLikelihoodPdf.h"
#include "goofit/Variable.h"

SumLikelihoodPdf::SumLikelihoodPdf (std::string n, const std::vector<PdfBase*> &comps)
  : GooPdf(0,n) 
{
  components = comps;

  std::vector<unsigned int> pindices;
  initialise(pindices); 
} 

__host__ fptype SumLikelihoodPdf::normalise () const {
  host_normalisation[parameters] = 1.0; 
  return 1; 
}

#include "SumPdf.h"
#ifdef NLL_CHECK
#include "GooStatsNLLCheck.h"
#endif
__host__ double SumLikelihoodPdf::sumOfNll (int ) const {
  double ret = 0;
#if defined(NLL_CHECK)
  GooStatsNLLCheck::get()->init("NLL_CHECK_gpu.root","gpu");
#endif
  for(unsigned int i = 0;i<components.size();++i) {
    ret += dynamic_cast<GooPdf*>(components.at(i))->calculateNLL();
#if defined(NLL_CHECK)
    GooStatsNLLCheck::get()->new_LL(ret);
#endif
  }
#if defined(NLL_CHECK)
  GooStatsNLLCheck::get()->record_finalLL(ret);
  GooStatsNLLCheck::get()->save();
  GooStatsNLLCheck::get()->print();
  std::cerr<<"Debug abort."<<std::endl;
  std::exit(0);
#endif
#if defined(RPF_CHECK) || defined(convolution_CHECK) || defined(NL_CHECK) || defined(spectrum_CHECK) || defined(Quenching_CHECK) || defined(Mask_CHECK)
  printf("final log(L) %.12le\n",ret);
  std::cerr<<"Debug abort."<<std::endl;
  std::exit(0);
#endif
  return ret; 
}
#include "DataPdf.h"
void SumLikelihoodPdf::fill_random() {
  for(unsigned int i = 0;i<components.size();++i) {
    DataPdf *pdf = dynamic_cast<DataPdf*>(components.at(i));
    if(pdf) pdf->fill_random();
  }
}
void SumLikelihoodPdf::fill_Asimov() {
  for(unsigned int i = 0;i<components.size();++i) {
    DataPdf *pdf = dynamic_cast<DataPdf*>(components.at(i));
    if(pdf) pdf->fill_Asimov();
  }
}
void SumLikelihoodPdf::cache() {
  for(unsigned int i = 0;i<components.size();++i) {
    DataPdf *pdf = dynamic_cast<DataPdf*>(components.at(i));
    if(pdf) pdf->cache();
  }
}
void SumLikelihoodPdf::restore() {
  for(unsigned int i = 0;i<components.size();++i) {
    DataPdf *pdf = dynamic_cast<DataPdf*>(components.at(i));
    if(pdf) pdf->restore();
  }
}
