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
#include "ConfigsetManager.h"
#include "DatasetManager.h"
#include <vector>
#include <memory>
class SumLikelihoodPdf;
#include "Module.h"
class InputManager : public Module {
  public:
    InputManager() : Module("InputManager") {};
    void setArgs(int _c,char **_v) { argc = _c; argv = _v; }
    InputManager(int _c,char **_v) : Module("InputManager"), argc(_c), argv(_v) {};
    virtual bool init() override;
  protected:
    int argc = -1;    ///< command line arguments
    char **argv = nullptr; ///< command line arguments

  public:
    void setInputBuilder(InputBuilder *);
    void setParSyncManager(ParSyncManager *);
    void setRawSpectrumProvider(RawSpectrumProvider *);
    virtual void initialize_configsets();
    virtual void fill_rawSpectrumProvider();
    virtual void create_variables();
    virtual void initialize_controllers();
    virtual void initialize_datasets();
    virtual void buildTotalPdf();
    std::map<DatasetManager*,std::unique_ptr<fptype []>> fillRandomData();
    std::map<DatasetManager*,std::unique_ptr<fptype []>> fillAsimovData();
    std::vector<ConfigsetManager*> Configsets();
    std::vector<DatasetManager*> Datasets();
    const std::vector<ConfigsetManager*> Configsets() const;
    const std::vector<DatasetManager*> Datasets() const;
    const OptionManager *GlobalOption() const;
    void setOutputFileName(const std::string &out) { outName = out; }
    const std::string &getOutputFileName() const { return outName; }
    SumLikelihoodPdf *getTotalPdf() { return totalPdf.get(); };
    const SumLikelihoodPdf *getTotalPdf() const { return totalPdf.get(); };
    void cachePars();
    void resetPars();
    RawSpectrumProvider *getProvider() const { return provider.get(); }
  protected:
    std::shared_ptr<InputBuilder> builder;
    std::shared_ptr<ParSyncManager> parManager;
    std::shared_ptr<SumLikelihoodPdf> totalPdf;
    std::shared_ptr<RawSpectrumProvider> provider;
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
    /**@}*/

    /**
     *  \defgroup Dataset part responsible for dataset
     *  @{
     */
  public:
    //! create list of datasets, controllers and associate them
    void registerDataset(DatasetManager *);
  protected:
    std::vector<std::shared_ptr<DatasetManager>> datasets;
    /**@}*/
};
#endif
