/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef MultiVariate_PDF_HH
#define MultiVariate_PDF_HH
#include "goofit/PDFs/GooPdf.h"
// S. Davini, “Measurement of the pep and CNO solar neutrino interaction rates in Borexino-I,” Springer International Publishing, Cham, 2013. page 96
// Equation (5.7) and (5.8)
class MultiVariatePdf : public GooPdf {
public:
  enum class MVLLType { StefanoDavini };
  MultiVariatePdf(std::string n, MVLLType MVLLtype, Variable *mv_var, BinnedDataSet *data,
                  const std::vector<BinnedDataSet *> &refs, const std::vector<GooPdf *> &pdf_0_,
                  const std::vector<GooPdf *> &pdf_1_, const std::vector<Variable *> &rate_0_,
                  const std::vector<Variable *> &rate_1_, int startbin_, int endbin_ /*startbin<=bin<endbin*/,
                  const SumPdf *sumpdf_, double binVolume_ = 1);
  __host__ fptype normalise() const;
  __host__ double calculateNLL() const;
  __host__ double sumOfNll(int numVars) const;

private:
  void copyTH1DToGPU(BinnedDataSet *data, const std::vector<BinnedDataSet *> &refs);
  void copyTH1DToGPU(BinnedDataSet *data, fptype &sum, fptype *dev_address[1], DEVICE_VECTOR<fptype> *&dev_vec_address);
  void calculate_m0m1() const;
  const std::vector<int> pdf_0;
  const std::vector<int> pdf_1;
  const std::vector<int> rate_0;
  const std::vector<int> rate_1;
  const std::vector<int> get_pdfids(const std::vector<GooPdf *> &pdfs);
  const std::vector<int> get_Nids(const std::vector<Variable *> &rates);
  double binVolume;
  const SumPdf *sumpdf;
  int MVid;
  fptype sum_k;
  fptype I0;
  fptype I1;
  int Nbin;
  int startbin;
  int endbin;
  static int totalPdf;
  fptype *dev_iConsts;
};
#endif
