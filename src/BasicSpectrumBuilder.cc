/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "BasicSpectrumBuilder.h"
#include "DatasetManager.h"
void BasicSpectrumBuilder::AddSiblings(ISpectrumBuilder *s) {
  siblings.push_back(std::shared_ptr<ISpectrumBuilder>(s));
}
GooPdf *BasicSpectrumBuilder::buildSpectrum(const std::string &name,DatasetManager *dataset) {
  const std::string &type = dataset->get<std::string>(name+"_type");
  for(auto sibling : siblings) {
    GooPdf *pdf = sibling->buildSpectrum(name,dataset);
    if(pdf) return pdf;
  }
  return nullptr;
}
