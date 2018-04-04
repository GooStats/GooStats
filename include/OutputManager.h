/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef OutputManager_H
#define OutputManager_H
class DatasetManager;
class BatchOutputManager;
class OutputHelper;
class OutputBuilder;
class PlotManager;
#include <memory>
#include <vector>
class InputManager;
class OutputManager {
  public:
    virtual bool init();
    virtual bool run();
    virtual bool finish();
    void adoptInputManager(InputManager *in) { inputManager = in; }
    void setBatchOutputManager(BatchOutputManager *);
    void setOutputBuilder(OutputBuilder *);
    void setPlotManager(PlotManager *);
  protected:
    InputManager *inputManager = nullptr;
    std::shared_ptr<BatchOutputManager> batchOut;
    std::shared_ptr<PlotManager> plot;
    std::shared_ptr<OutputHelper> outputHelper;
    std::shared_ptr<OutputBuilder> outputBuilder;
};
#endif
