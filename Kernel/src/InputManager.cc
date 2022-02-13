/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "InputManager.h"
#include "InputBuilder.h"
#include "ParSyncManager.h"
#include "RawSpectrumProvider.h"
#include "goofit/Variable.h"
#include "GooStatsException.h"
#include "SumLikelihoodPdf.h"
bool InputManager::init() {
  if(!parManager) {
    std::cerr<<"Warning: ParSyncManager not set, default EStrategy are used."<<std::endl;
    setParSyncManager(new ParSyncManager());
  }
  BasicManager::setParSyncConfig(parManager->getStrategies());
  if(!provider) {
    std::cerr<<"Warning: RawSpectrumProvider not set, default EStrategy are used."<<std::endl;
    setRawSpectrumProvider(new RawSpectrumProvider());
  }
  if(!builder) {
    std::cerr<<"Error: InputBuilder not set, please set it before calling "
              "InputManager::init"<<std::endl;
    throw GooStatsException("InputBuilder not set in InputManager");
  }
  initializeConfigsets();
  outName = builder->loadOutputFileName(argc, argv, Configsets());
  fillRawSpectrumProvider();
  initializeDatasets();
  totalPdf = std::shared_ptr<SumLikelihoodPdf>(builder->buildTotalPdf(Datasets()));
  cachePars();
  return true;
}
void InputManager::setInputBuilder(InputBuilder *input) {
  builder = std::shared_ptr<InputBuilder>(input);
}

void InputManager::setParSyncManager(ParSyncManager *par) {
  parManager = std::shared_ptr<ParSyncManager>(par);
  parManager->init();
  BasicManager::setParSyncConfig(parManager->getStrategies());
}
void InputManager::setRawSpectrumProvider(RawSpectrumProvider *p) {
  provider = std::shared_ptr<RawSpectrumProvider>(p);
}

void InputManager::initializeConfigsets() {
  auto configs = builder->buildConfigsetManagers(parManager.get(), argc, argv);
  if(configs.empty()) throw GooStatsException("No configset found");
  for(auto configset : configs) {
    registerConfigset(configset);
    configset->printAllOptions();
    builder->createVariables(configset);
  }
  BasicManager::dump();
}

void InputManager::fillRawSpectrumProvider() {
  for(const auto& configset: configsets)
    builder->fillRawSpectrumProvider(provider.get(),configset.get());
}

void InputManager::initializeDatasets() {
  for(auto config : configsets) {
    auto controllers = builder->buildDatasetsControllers(config.get());
    for(auto controller : controllers) {
      auto dataset = controller->createDataset();
      this->registerDataset(dataset);
      controller->collectInputs(dataset);
      if(dataset->has<std::vector<std::string>>("components")) {
        builder->fillDataSpectra(dataset,provider.get());
        builder->buildRawSpectra(dataset,provider.get());
        builder->buildComponenets(dataset,provider.get());
      }
      controller->buildLikelihood(dataset);
    }
  }
}
void InputManager::fillRandomData() {
  getTotalPdf()->fill_random();
}
void InputManager::fillAsimovData() {
  getTotalPdf()->fill_Asimov();
}
std::vector<ConfigsetManager*> InputManager::Configsets() {
  std::vector<ConfigsetManager*> configsets_;
  for(auto configset: configsets) {
    configsets_.push_back(configset.get());
  }
  return configsets_;
}
std::vector<DatasetManager*> InputManager::Datasets() {
  std::vector<DatasetManager*> datasets_;
  for(auto dataset: datasets) {
    datasets_.push_back(dataset.get());
  }
  return datasets_;
}
const std::vector<ConfigsetManager*> InputManager::Configsets() const {
  std::vector<ConfigsetManager*> configsets_;
  for(auto configset: configsets) {
    configsets_.push_back(configset.get());
  }
  return configsets_;
}
const std::vector<DatasetManager*> InputManager::Datasets() const {
  std::vector<DatasetManager*> datasets_;
  for(auto dataset: datasets) {
    datasets_.push_back(dataset.get());
  }
  return datasets_;
}
const OptionManager *InputManager::GlobalOption() const { 
  return static_cast<OptionManager*>(configsets.front().get()); 
}
void InputManager::cachePars() {
  std::vector<Variable*> pars;
  getTotalPdf()->getParameters(pars);
  cachedParsInit.clear();
  cachedParsErr.clear();
  cachedParsUL.clear();
  cachedParsLL.clear();
  cachedParsFix.clear();
  for(auto par : pars) {
    cachedParsInit.push_back(par->value);
    cachedParsErr.push_back(par->error);
    cachedParsUL.push_back(par->upperlimit);
    cachedParsLL.push_back(par->lowerlimit);
    cachedParsFix.push_back(par->fixed);
  }
}
void InputManager::resetPars() {
  std::vector<Variable*> pars;
  getTotalPdf()->getParameters(pars);
  for(size_t i = 0;i<pars.size();++i) {
    pars.at(i)->value = cachedParsInit.at(i);
    pars.at(i)->error = cachedParsErr.at(i);
    pars.at(i)->upperlimit = cachedParsUL.at(i);
    pars.at(i)->lowerlimit = cachedParsLL.at(i);
    pars.at(i)->fixed = cachedParsFix.at(i);
  }
}
void InputManager::registerConfigset(ConfigsetManager *configset) { 
  std::cout<<"InputManager::registerConfigset("<<configset->name()<<")"<<std::endl;
  configsets.push_back(std::shared_ptr<ConfigsetManager>(configset)); 
}
void InputManager::registerDataset(DatasetManager *dataset) { 
  std::cout<<"InputManager::registerDataset("<<dataset->name()<<")"<<std::endl;
  datasets.push_back(std::shared_ptr<DatasetManager>(dataset)); 
}
