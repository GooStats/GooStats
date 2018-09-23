/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
/*! \class InputBuilder
 *  \brief builder class used by InputManager
 *
 *   This is a utlity class and is responsible for building the Configset 
 */
#ifndef InputBuilder_H
#define InputBuilder_H
class GooPdf;
#include <string>
#include <vector>
#include <memory>
class OptionManager;
class ConfigsetManager;
class DatasetController;
class DatasetManager;
class InputConfig;
class ParSyncManager;
class RawSpectrumProvider;
class ISpectrumBuilder;
class InputBuilder {
  public:
    //! load the name of output file from command-line args.
    virtual std::string loadOutputFileNameFromCmdArgs(int,char **) = 0;
    //! load number of configs / location of configuration files from command-line args.
    virtual std::vector<InputConfig *> loadConfigsFromCmdArgs(int argc,char **argv) = 0;
    //! construct and fill the IDataManager part
    virtual ConfigsetManager *buildConfigset(ParSyncManager *parManager,const InputConfig &config) = 0;
    //! fill raw spectrum providers
    virtual void fillRawSpectrumProvider(RawSpectrumProvider *,ConfigsetManager*) = 0;
    //! set-up config-set level parameters
    virtual bool configParameters(ConfigsetManager *configset) = 0;
    //! install spectrum type hanlder
    virtual bool installSpectrumBuilder(ISpectrumBuilder *) = 0;
    //! build sets of datasetcontroller. each controller correspond to a spectrum
    virtual std::vector<std::shared_ptr<DatasetController>> buildDatasetsControllers(ConfigsetManager *configset) = 0;
    //! construct a dataset manager based on a datasetcontroller
    virtual DatasetManager *buildDataset(DatasetController *) = 0;
    //! build the raw spectra used for convolution
    virtual bool buildRawSpectra(DatasetManager *dataset,RawSpectrumProvider *provider) = 0;
    //! build the components of datasetmanager
    virtual bool buildComponenets(DatasetManager *,RawSpectrumProvider *provider) = 0;
    //! set-up data-set level parameters
    virtual bool configParameters(DatasetManager *) = 0;
    //! provide the option manager
    virtual OptionManager *createOptionManager() = 0;
    //! initialize the OptionManager part of ConfigsetManager, and parse the config file
    virtual bool fillOptions(ConfigsetManager *configset,const std::string &configFileName,OptionManager* ) __attribute__ ((deprecated)) { 
      return fillOptions(configset,configFileName); // to be compatible with bx-GooStats
    }
    virtual bool fillOptions(ConfigsetManager *,const std::string &) { return true; }; // to be compatible with bx-GooStats
    virtual bool fillOptions(ConfigsetManager *,int ,char **) { return true; }; // to be compatible with bx-GooStats
    //! build the total pdf from the datasets
    virtual GooPdf *buildTotalPdf(const std::vector<DatasetManager*> &) = 0;
};
#endif
