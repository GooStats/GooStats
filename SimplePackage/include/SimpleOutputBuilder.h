/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
/*! \class SimpleOutputBuilder
 *  \brief builder class used by OutputManager
 *
 *   This is a utlity class and is responsible for building the Configset 
 */
#ifndef SimpleOutputBuilder_H
#define SimpleOutputBuilder_H
#include "OutputBuilder.h"
#include <map>
#include <string>
class SimpleOutputBuilder : public OutputBuilder {
  public:
    //! load number of configs / location of configuration files from command-line args.
    void registerOutputTerms(OutputHelper *, InputManager *,GSFitManager *) override;
    void bindAllParameters(BatchOutputManager *,OutputHelper*) override;
    void fillAllParameters(BatchOutputManager *,OutputHelper*) override;
    void flushOstream(BatchOutputManager *,OutputHelper *,std::ostream &) override;
    void draw(int event,GSFitManager *,PlotManager *,InputManager *) override;
  private:
    std::map<std::string,double> goodness;
};
#endif
