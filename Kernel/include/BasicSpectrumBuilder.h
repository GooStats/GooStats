/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
/*! \class IBasicSpectrumBuilder
 *  \brief protocal for spectrum builder
 *
 *     It will be used to construct spectrum in InputBuilder
 */
#ifndef BasicSpectrumBuilder_H
#define BasicSpectrumBuilder_H
#include <memory>
#include <vector>

#include "ISpectrumBuilder.h"
class BasicSpectrumBuilder : public ISpectrumBuilder {
 public:
  GooPdf *buildSpectrum(const std::string &, DatasetManager *) override;
  //! build the spectrum of name for the dataset
  void AddSiblings(ISpectrumBuilder *) final;

 private:
  std::vector<std::shared_ptr<ISpectrumBuilder>> siblings;
};
#endif
