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
    std::cout<<"Warning: ParSyncManager not set, default strategy are used."<<std::endl;
    setParSyncManager(new ParSyncManager());
  }
  if(!provider) {
    std::cout<<"Warning: RawSpectrumProvider not set, default strategy are used."<<std::endl;
    setRawSpectrumProvider(new RawSpectrumProvider());
  }
  if(!builder) {
    std::cout<<"Error: InputBuilder not set, please set it before calling InputManager::init"<<std::endl;
    throw GooStatsException("InputBuilder not set in InputManager");
  }
  outName = builder->loadOutputFileNameFromCmdArgs(argc,argv);
  initialize_configsets();
  fill_rawSpectrumProvider();
  create_variables();
  initialize_controllers();
  initialize_datasets();
  buildTotalPdf();
  cachePars();
  return true;
}
void InputManager::setInputBuilder(InputBuilder *input) {
  builder = std::shared_ptr<InputBuilder>(input);
}

void InputManager::setParSyncManager(ParSyncManager *par) {
  parManager = std::shared_ptr<ParSyncManager>(par);
  parManager->init();
  BasicManager::setStrategyManager(parManager.get());
}
void InputManager::setRawSpectrumProvider(RawSpectrumProvider *p) {
  provider = std::shared_ptr<RawSpectrumProvider>(p);
}
void InputManager::initialize_configsets() {
  // step 1: load number of configs / location of configuration files from command-line args.
  auto configs = builder->loadConfigsFromCmdArgs(argc,argv);
  if(!configs.size()) throw GooStatsException("No configset found");
  for(auto config : configs) {
    // step 2: for each config set, construct (empty) config objects
    auto configset = builder->buildConfigset(parManager.get(),*config);
    registerConfigset(configset);
    // step 3: populate options
    builder->fillOptions(configset,config->configFile);
    builder->fillOptions(configset,argc,argv);
    configset->printAllOptions();
    // step 4: configure parameters of the configset
    builder->configParameters(configset);
  }
}
void InputManager::initialize_controllers() {
  for(auto config : configsets) {
    auto controllers = builder->buildDatasetsControllers(config.get());
    config->setDatasetControllers(controllers);
    for(auto controller : config->getDatasetControllers()) {
      auto dataset = controller->createDataset();
      dataset->setDelegate(controller.get());
      this->registerDataset(dataset);
    }
  }
}
void InputManager::create_variables() {
  std::cout<<"Dumping pre-created rates:"<<std::endl;
  for(auto configset: configsets) {
    builder->createVariables(configset.get());
    configset->dump(configset->name()+">");
  }
}
void InputManager::fill_rawSpectrumProvider() {
  for(auto configset: configsets)
    builder->fillRawSpectrumProvider(provider.get(),configset.get());
}
void InputManager::initialize_datasets() {
  for(auto dataset : datasets) {
    dataset->initialize();
    if(dataset->has<std::vector<std::string>>("components")) {
      builder->fillDataSpectra(dataset.get(),provider.get());
      builder->buildRawSpectra(dataset.get(),provider.get());
      builder->buildComponenets(dataset.get(),provider.get());
    }
    builder->configParameters(dataset.get());
    dataset->buildLikelihood();
  }
}
void InputManager::buildTotalPdf() {
  totalPdf = std::shared_ptr<SumLikelihoodPdf>(builder->buildTotalPdf(Datasets()));
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
