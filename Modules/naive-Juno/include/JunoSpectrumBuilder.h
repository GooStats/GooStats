/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
/*! \class JunoSpectrumBuilder
 *  \brief example builder class used by InputManager
 *
 *   This is a utlity class and is responsible for building the Configset 
 */
#ifndef JunoSpectrumBuilder_H
#define JunoSpectrumBuilder_H
#include "BasicSpectrumBuilder.h"
class JunoSpectrumBuilder : public BasicSpectrumBuilder {
  public:
    GooPdf *buildSpectrum(const std::string &, DatasetManager *) final;
  private:
    SpectrumBuilderFun buildReactor;
    SpectrumBuilderFun buildOscillatedReactor;
};
#endif
