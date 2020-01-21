/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
/*! \class OutputBuilder
 *  \brief builder class used by OutputManager
 *
 *   This is a utlity class and is responsible for building the Configset 
 */
#ifndef OutputBuilder_H
#define OutputBuilder_H
#include <ostream>
class OutputHelper;
class InputManager;
class BatchOutputManager;
class PlotManager;
class GSFitManager;
class OutputBuilder {
  public:
    virtual ~OutputBuilder() {};
    //! load number of configs / location of configuration files from command-line args.
    virtual void registerOutputTerms(OutputHelper *, InputManager *,GSFitManager *) = 0;
    virtual void bindAllParameters(BatchOutputManager *,OutputHelper*) = 0;
    virtual void fillAllParameters(BatchOutputManager *,OutputHelper*) = 0;
    virtual void flushOstream(BatchOutputManager *,OutputHelper *,std::ostream &) = 0;
    virtual void draw(int event,GSFitManager*,PlotManager *,InputManager *) = 0;
};
#endif
