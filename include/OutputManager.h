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
#include "Module.h"
class TFile;
class InputManager;
class OutputManager : public Module {
  public:
    OutputManager() : Module("OutputManager") { }
    virtual bool preinit() override;
    virtual bool init() override;
    virtual bool run(int event) override;
    virtual bool finish() override;
    virtual bool postfinish() override;
  private:
    InputManager *getInputManager();
  public:
    void setOutputFile(const std::string &fname);
    TFile *getOutputFile() { return file; }
    void setBatchOutputManager(BatchOutputManager *);
    void setOutputBuilder(OutputBuilder *);
    void setPlotManager(PlotManager *);
    BatchOutputManager *getBatchOutputManager() const { return batchOut.get(); }
    void subFit(int event);
  protected:
    std::shared_ptr<OutputHelper> outputHelper;
    std::shared_ptr<OutputBuilder> outputBuilder;
    std::shared_ptr<BatchOutputManager> batchOut;
    std::shared_ptr<PlotManager> plot;
    TFile *file;
};
#endif
