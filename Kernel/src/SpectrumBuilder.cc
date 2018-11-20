/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "SpectrumBuilder.h"
#include "GooStatsException.h"
#include "DatasetManager.h"
GooPdf *SpectrumBuilder::buildSpectrum(const std::string &name,DatasetManager *dataset) {
  GooPdf *pdf = this->BasicSpectrumBuilder::buildSpectrum(name,dataset);
  if(pdf) return pdf;
  const std::string &type = dataset->get<std::string>(name+"_type");
  throw GooStatsException("Cannot build spectrum ["+name+"] unknown/empty type ["+type+"]");
}
