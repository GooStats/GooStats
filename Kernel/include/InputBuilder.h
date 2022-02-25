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
class SumLikelihoodPdf;
#include "IOptionParser.h"
#include <memory>
#include <string>
#include <vector>
class ConfigsetManager;
class DatasetController;
class DatasetManager;
class IOptionParser;
class ParSyncManager;
class RawSpectrumProvider;
class ISpectrumBuilder;
class PdfBase;
class InputBuilder {
  public:
  virtual ~InputBuilder(){};
  //! load the name of output file from command-line args.
  virtual std::string loadOutputFileName(int, const char **, std::vector<ConfigsetManager *> configsets = {}) = 0;
  //! load number of configs / location of configuration files from command-line args.
  virtual std::vector<ConfigsetManager *> buildConfigsetManagers(ParSyncManager *parManager, int argc,
                                                                 const char **argv) = 0;
  //! fill raw spectrum providers
  virtual void fillRawSpectrumProvider(RawSpectrumProvider *, ConfigsetManager *) = 0;
  //! create list of vars, so DatasetManager can call ConfigManager::var(name)
  virtual void createVariables(ConfigsetManager *) = 0;
  //! install spectrum type hanlder
  virtual bool installSpectrumBuilder(ISpectrumBuilder *, RawSpectrumProvider *provider) = 0;
  //! build sets of datasetcontroller. each controller correspond to a spectrum
  virtual std::vector<std::shared_ptr<DatasetController>> buildDatasetsControllers(ConfigsetManager *configset) = 0;
  //! fill data spectra
  virtual bool fillDataSpectra(DatasetManager *dataset, RawSpectrumProvider *provider) = 0;
  //! build the raw spectra used for convolution
  virtual PdfBase *recursiveBuild(const std::string &name, DatasetManager *, RawSpectrumProvider *,
                              ISpectrumBuilder *) = 0;
  //! build the components of datasetmanager
  virtual bool buildComponenets(DatasetManager *, RawSpectrumProvider *provider, ISpectrumBuilder *) = 0;
  //! build the total pdf from the datasets
  virtual SumLikelihoodPdf *buildTotalPdf(const std::vector<DatasetManager *> &) = 0;
};
#endif
