/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
/*! \class solarB8SpectrumBuilder
 *  \brief example builder class used by InputManager
 *
 *   This is a utlity class and is responsible for building the Configset 
 */
#ifndef solarB8SpectrumBuilder_H
#define solarB8SpectrumBuilder_H
#include "BasicSpectrumBuilder.h"
class solarB8SpectrumBuilder : public BasicSpectrumBuilder {
  public:
    GooPdf *buildSpectrum(const std::string &, DatasetManager *) final;
  private:
    SpectrumBuilderFun buildsolarB8;
    SpectrumBuilderFun buildOscillatedsolarB8;
  private:
    GooPdf *_buildOscillatedsolarB8(const std::string &name,DatasetManager *dataset,bool oscOn);
};
#endif
