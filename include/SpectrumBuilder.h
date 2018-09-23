/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
/*! \class SpectrumBuilder
 *  \brief example builder class used by InputManager
 *
 *   This is a utlity class and is responsible for building the Configset 
 */
#ifndef SpectrumBuilder_H
#define SpectrumBuilder_H
#include "BasicSpectrumBuilder.h"
#include <vector>
#include <memory>
class RawSpectrumProvider;
class Variable;
class BinnedDataSet;
class SpectrumBuilder : public BasicSpectrumBuilder {
  public:
    SpectrumBuilder(RawSpectrumProvider *_p) : provider(_p) { };
    SpectrumBuilder() = delete;
    GooPdf *buildSpectrum(const std::string &,DatasetManager*) final;
    BinnedDataSet *loadRawSpectrum(Variable *x,const std::string &);
  private:
    SpectrumBuilderFun buildMC;
    SpectrumBuilderFun buildAna;
    SpectrumBuilderFun buildAnaShifted;
    SpectrumBuilderFun buildAnaPeak;
    SpectrumBuilderFun buildTODO;
    SpectrumBuilderFun buildAnaBasic;
    RawSpectrumProvider *provider;
};
#endif
