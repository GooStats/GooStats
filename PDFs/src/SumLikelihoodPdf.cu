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

int total_sumllpdf = 0;
SumLikelihoodPdf::SumLikelihoodPdf (std::string n, const std::vector<PdfBase*> &comps)
  : GooPdf(0,n) 
{
  assert(total_sumllpdf++ == 0);

  components = comps;

  std::vector<unsigned int> pindices;
  initialise(pindices); 
} 

__host__ fptype SumLikelihoodPdf::normalise () const {
  host_normalisation[parameters] = 1.0; 
  return 1; 
}

#include "SumPdf.h"
__host__ double SumLikelihoodPdf::sumOfNll (int ) const {
  double ret = 0;
  for(unsigned int i = 0;i<components.size();++i) {
    ret += dynamic_cast<GooPdf*>(components.at(i))->calculateNLL();
  }
#if defined(NLL_CHECK) || defined(RPF_CHECK) || defined(convolution_CHECK) || defined(NL_CHECK) || defined(spectrum_CHECK) || defined(Quenching_CHECK) || defined(Mask_CHECK)
  std::cerr<<"Debug abort."<<std::endl;
  std::abort();
#endif
  return ret; 
}
#include "SumPdf.h"
void SumLikelihoodPdf::fill_random() {
  for(unsigned int i = 0;i<components.size();++i) {
    SumPdf *sum_term = dynamic_cast<SumPdf*>(components.at(i));
    if(sum_term) sum_term->fill_random();
  }
}
