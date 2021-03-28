/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef DUMPER_PDF_HH
#define DUMPER_PDF_HH
#include "goofit/PDFs/GooPdf.h"
#include "goofit/Variable.h"

template<class Base>
class DumperPdf : public Base {
  public:
    DumperPdf(Base *base) : Base(*base) {};
    void dumpPdf();
    void dumpPdf(BinnedDataSet *data);
    void dumpPdf(const std::vector<BinnedDataSet*> &data);
    void dumpConvolution(Variable *var);
    void dumpIndices();
    virtual ~DumperPdf() {};
};
template<class Base>
void DumperPdf<Base>::dumpPdf() {
  Variable *var = *(this->obsBegin());
  std::vector<GooPdf*> pdfs;
  std::vector<std::vector<fptype> > pdf_values;

  std::vector<fptype> pdf_value;
  this->evaluateAtPoints(var,pdf_value);
  pdfs.push_back(this);
  pdf_values.push_back(pdf_value);
  double step = (var->upperlimit - var->lowerlimit) / var->numbins;
  printf("Now canning val\n");
  for(size_t i = 0;i<pdf_value.size();++i) {
    var->value = var->lowerlimit + (i+0.5)*step;
    printf("%s %.10le => ",var->name.c_str(), var->value);
    for(size_t j = 0;j<pdfs.size();++j)
      printf(" %s %.10le",this->getName().c_str(),pdf_values.at(j).at(i));
    printf("\n");
  }
}
#endif
