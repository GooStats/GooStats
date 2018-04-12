/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef SUM_PDF_HH
#define SUM_PDF_HH

#include <map>
#include "goofit/PDFs/GooPdf.h"
template<class T>
class DumperPdf;
class SumLikelihoodPdf;
class BinnedDataSet;
#include <memory>
#define NPDFSIZE_SumPdf 500
extern MEM_CONSTANT fptype* dev_componentWorkSpace[NPDFSIZE_SumPdf];
extern DEVICE_VECTOR<fptype>* componentWorkSpace[NPDFSIZE_SumPdf];

class SumPdf : public GooPdf {
public:

  SumPdf (std::string n, const fptype norm_,const std::vector<Variable*> &, const std::vector<PdfBase*> &comps, Variable *npe);
  SumPdf (std::string n, const std::vector<PdfBase*> &comps, const std::vector<const BinnedDataSet*> &mask,Variable *npe);
  SumPdf (std::string n, const std::vector<PdfBase*> &comps, Variable *npe);
  __host__ virtual fptype normalise () const;
  const std::vector<PdfBase*> &Components() const { return components; }
  const std::vector<Variable*> &Weights() const { return _weights; }
  BinnedDataSet *getData();
  double Norm() const { return norm; }
  double Chi2() ;
  int NDF() ;
  static int registerFunc(PdfBase *pdf);
  std::unique_ptr<fptype []> fill_random();

protected:
  void register_components(const std::vector<PdfBase*> &comps,int N);
  void set_startstep(fptype norm);
  __host__ void copyHistogramToDevice (const std::vector<const BinnedDataSet*> &masks);
  __host__ void copyHistogramToDevice (const BinnedDataSet* mask,int id);
  static int registerMask(BinnedDataSet *mask);
  static int maskId;
  static std::map<BinnedDataSet*,int> maskmap;
  bool updated() const { return m_updated; }
#ifdef NLL_CHECK
  __host__ double sumOfNll (int numVars) const;
  friend class DarkNoiseConvolutionPdf;
#endif
  std::vector<unsigned int> pindices;
  fptype* dev_iConsts; 
  int workSpaceIndex;
  bool extended; 
  fptype norm;
  static std::map<PdfBase*,int> funMap;
  mutable bool m_updated;
  thrust::host_vector<fptype> cached_sumV;
  std::vector<Variable*> _weights;
  BinnedDataSet *dataset;
  friend class DumperPdf<SumLikelihoodPdf>;
};

#endif
