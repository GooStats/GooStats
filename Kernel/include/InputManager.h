/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
/*! \class InputManager
 *  \brief class responsible for collecting and preparing inputs
 *
 *   This class will collect raw spectrum, parse configurations, setup
 *   parameters and build the objects recognizable for GooFit
 */
#ifndef InputManager_H
#define InputManager_H
class InputBuilder;
class ParSyncManager;
class RawSpectrumProvider;
#include <memory>
#include <vector>

#include "ConfigsetManager.h"
#include "DatasetManager.h"
#include "ISpectrumBuilder.h"
class SumLikelihoodPdf;
#include "Module.h"
class InputManager : public Module {
 public:
  InputManager(int _c, const char **_v) : Module("InputManager"), argc(_c), argv(_v){};
  bool init() override;

 protected:
  const int argc;     ///< command line arguments
  const char **argv;  ///< command line arguments

 public:
  void setInputBuilder(InputBuilder *);
  void setParSyncManager(ParSyncManager *);
  void setRawSpectrumProvider(RawSpectrumProvider *);
  virtual void initializeConfigsets();
  virtual void fillRawSpectrumProvider();
  virtual void initializeDatasets();
  void fillRandomData();
  void fillAsimovData();
  std::vector<ConfigsetManager *> Configsets();
  [[nodiscard]] std::vector<ConfigsetManager *> Configsets() const;
  std::vector<DatasetController *> Controllers();
  std::vector<DatasetManager *> Datasets();
  [[nodiscard]] const OptionManager *GlobalOption() const { return globalConfigset.get(); }
  void setOutputFileName(const std::string &out) { outName = out; }
  [[nodiscard]] const std::string &getOutputFileName() const { return outName; }
  SumLikelihoodPdf *getTotalPdf() { return totalPdf.get(); };
  [[nodiscard]] const SumLikelihoodPdf *getTotalPdf() const { return totalPdf.get(); };
  void cachePars();
  void resetPars();
  [[nodiscard]] RawSpectrumProvider *getProvider() const { return provider.get(); }

 protected:
  std::shared_ptr<InputBuilder> builder;
  std::shared_ptr<ParSyncManager> parManager;
  std::shared_ptr<SumLikelihoodPdf> totalPdf;
  std::shared_ptr<RawSpectrumProvider> provider;
  std::shared_ptr<ISpectrumBuilder> spcBuilder;
  std::string outName;

 private:
  std::vector<double> cachedParsInit;
  std::vector<double> cachedParsErr;
  std::vector<double> cachedParsUL;
  std::vector<double> cachedParsLL;
  std::vector<bool> cachedParsFix;

  /**
     *  \defgroup Dataset part responsible for dataset
     *  @{
     */
 public:
  void registerConfigset(ConfigsetManager *);

 protected:
  //! Config-set is the minimul data-set unit in GooStats.
  //! One config-set can contain multiple spectrum, but they
  //! In GooFit there is a even smaller unit
  std::vector<std::shared_ptr<ConfigsetManager>> configsets;
  std::shared_ptr<ConfigsetManager> globalConfigset;
  /**@}*/

  /**
     *  \defgroup Dataset part responsible for dataset
     *  @{
     */
 public:
  //! create list of datasets, controllers and associate them
  void registerController(std::shared_ptr<DatasetController> controller);

 protected:
  std::vector<std::shared_ptr<DatasetController>> controllers;
  /**@}*/
};
#endif
