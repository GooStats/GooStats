#ifndef SUM_PDF_HH
#define SUM_PDF_HH

#include <map>
#include "goofit/PDFs/GooPdf.h"
template<class T>
class DumperPdf;
class SumLikelihoodPdf;
class BinnedDataSet;

class SumPdf : public GooPdf {
public:

  SumPdf (std::string n, const fptype norm_,const std::vector<Variable*> &weights, const std::vector<PdfBase*> &comps, Variable *npe);
  SumPdf (std::string n, const std::vector<PdfBase*> &comps, const std::vector<const BinnedDataSet*> &mask,Variable *npe);
  SumPdf (std::string n, const std::vector<PdfBase*> &comps, Variable *npe);
  __host__ virtual fptype normalise () const;
  const std::vector<PdfBase*> &Components() const { return components; }
  double Norm() const { return norm; }
  static int registerFunc(PdfBase *pdf);
  void fill_random();

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
  friend class DumperPdf<SumLikelihoodPdf>;
};

#endif
