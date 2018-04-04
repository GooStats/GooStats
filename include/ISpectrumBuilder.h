/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
/*! \class IISpectrumBuilder
 *  \brief protocal for spectrum builder
 *
 *     It will be used to construct spectrum in InputBuilder
 */
#ifndef ISpectrumBuilder_H
#define ISpectrumBuilder_H
class GooPdf;
#include <string>
class DatasetManager;
class ISpectrumBuilder {
  public:
    //! build the spectrum of name for the dataset
    virtual void AddSiblings(ISpectrumBuilder *) = 0;
    virtual GooPdf *buildSpectrum(const std::string &, DatasetManager *) = 0;
    //! build a specific type of spectrum
    typedef GooPdf *(SpectrumBuilderFun)(const std::string &, DatasetManager*);
};
#endif
