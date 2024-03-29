/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
/*! \class SimpleInputBuilder
 *  \brief example builder class used by InputManager
 *
 *   This is a utlity class and is responsible for building the Configset
 */
#ifndef SimpleInputBuilder_H
#define SimpleInputBuilder_H
#include <cstdlib>

#include "InputBuilder.h"
struct InputConfig;
#include <memory>
class BasicSpectrumBuilder;
class SimpleInputBuilder : public InputBuilder {
 public:
  //! load the name of output file from command-line args.
  std::string loadOutputFileName(int argc, const char **argv, std::vector<ConfigsetManager *> configsets) override;
  //! load number of configs / location of configuration files from command-line args.
  //! here we use pointer to allow polymorphism. Better design would use template.
  std::pair<ConfigsetManager *, std::vector<ConfigsetManager *>> buildConfigsetManagers(ParSyncManager *parManager,
                                                                                        int argc,
                                                                                        const char **argv) override;
  //! fill raw spectrum providers
  void fillRawSpectrumProvider(RawSpectrumProvider *, ConfigsetManager *) override;
  //! create list of vars, so DatasetManager can call ConfigManager::var(name)
  void createVariables(ConfigsetManager *) override;
  //! install spectrum type hanlder
  bool installSpectrumBuilder(ISpectrumBuilder *builder, RawSpectrumProvider *provider) override;
  //! build sets of datasetcontroller. each controller correspond to a spectrum
  std::vector<std::shared_ptr<DatasetController>> buildDatasetsControllers(ConfigsetManager *configset) override;
  //! fill data spectra
  bool fillDataSpectra(DatasetManager *, RawSpectrumProvider *) override;
  //! build the raw spectra used for convolution
  PdfBase *recursiveBuild(const std::string &name,
                          DatasetManager *dataset,
                          RawSpectrumProvider *provider,
                          ISpectrumBuilder *spcBuilder) override;
  //! build the components of datasetmanager
  bool buildComponenets(DatasetManager *dataset, RawSpectrumProvider *provider, ISpectrumBuilder *spcBuilder) override;
  //! build the total pdf from the datasets
  SumLikelihoodPdf *buildTotalPdf(const std::vector<DatasetManager *> &) override;

 private:
  void usage(const char *const *argv) const;
};
#endif
