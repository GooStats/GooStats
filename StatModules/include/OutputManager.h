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
#include "StatModule.h"
class TFile;
class InputManager;
class OutputManager : public StatModule {
  public:
    OutputManager(const std::string &_name="OutputManager") : StatModule(_name) { }
    virtual bool preinit() override;
    virtual bool init() override;
    virtual bool run(int event) override;
    virtual bool finish() override;
    virtual bool postfinish() override;
  public:
    void setOutputFile(const std::string &fname);
    TFile *getOutputFile() { return file; }
    void setBatchOutputManager(BatchOutputManager *);
    void setOutputBuilder(OutputBuilder *);
    void setPlotManager(PlotManager *);
    BatchOutputManager *getBatchOutputManager() const { return batchOut.get(); }
    void subFit(int event);
    const OutputHelper *getOutputHelper() const { return outputHelper.get(); }
  protected:
    std::shared_ptr<OutputHelper> outputHelper;
    std::shared_ptr<OutputBuilder> outputBuilder;
    std::shared_ptr<BatchOutputManager> batchOut;
    std::shared_ptr<PlotManager> plot;
    TFile *file;
};
#endif
