/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
/*! \class ReactorSpectrumBuilder
 *  \brief example builder class used by InputManager
 *
 *   This is a utlity class and is responsible for building the Configset 
 */
#ifndef ReactorSpectrumBuilder_H
#define ReactorSpectrumBuilder_H
#include "BasicSpectrumBuilder.h"
class ReactorSpectrumBuilder : public BasicSpectrumBuilder {
  public:
    GooPdf *buildSpectrum(const std::string &, DatasetManager *) final;
  private:
    SpectrumBuilderFun buildReactor;
    SpectrumBuilderFun buildOscillatedReactor;
  private:
    GooPdf *_buildOscillatedReactor(const std::string &name,DatasetManager *dataset,bool oscOn);
};
#endif
