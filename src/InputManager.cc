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
#include "goofit/BinnedDataSet.h"
#include "SumPdf.h"
#include "GooStatsException.h"
bool InputManager::init() {
  outName = builder->loadOutputFileNameFromCmdArgs(argc,argv);
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
  return true;
}
bool InputManager::run() {
  return true;
}
bool InputManager::finish() {
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
  for(auto config : configs) {
    // step 2: for each config set, construct (empty) config objects
    auto configset = builder->buildConfigset(parManager.get(),*config);
    registerConfigset(configset);
    // step 3: populate options
    builder->fillOptions(configset,config->configFile,builder->createOptionManager());
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
void InputManager::fill_rawSpectrumProvider() {
  for(auto configset: configsets)
    builder->fillRawSpectrumProvider(provider.get(),configset.get());
}
void InputManager::initialize_datasets() {
  for(auto dataset : datasets) {
    dataset->initialize();
    builder->buildRawSpectra(dataset.get(),provider.get());
    builder->buildComponenets(dataset.get(),provider.get());
    builder->configParameters(dataset.get());
    dataset->buildLikelihood();
  }
}
void InputManager::buildTotalPdf() {
  totalPdf = std::shared_ptr<GooPdf>(builder->buildTotalPdf(Datasets()));
}
std::map<DatasetManager*,std::unique_ptr<fptype []>> InputManager::fillRandomData() {
  std::map<DatasetManager*,std::unique_ptr<fptype []>> datas;
  for(auto dataset: datasets) {
    SumPdf *sumpdf = dynamic_cast<SumPdf*>(dataset->getLikelihood());
    if(!sumpdf) continue;
    Variable *Evis = dataset->get<Variable*>("Evis");
    BinnedDataSet* data= new BinnedDataSet(Evis);
    sumpdf->setData(data);
    std::unique_ptr<fptype []> res = sumpdf->fill_random();
    for(int i = 0;i<Evis->numbins;++i) {
      data->setBinContent(i,res[i*3+1]);
    }
    datas.insert(std::make_pair(dataset.get(),std::move(res)));
  }
  return datas;
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
