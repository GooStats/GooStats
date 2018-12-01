/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "SimpleInputBuilder.h"
#include "InputConfig.h"
#include <iostream>
#include "GooStatsException.h"
#include "RawSpectrumProvider.h"
#include "ParSyncManager.h"
#include "ConfigsetManager.h"
#include "Utility.h"
#include <fstream>
#include "DatasetController.h"
#include "SimpleSpectrumBuilder.h"
#include "SimpleDatasetController.h"
#include "PullDatasetController.h"
#include "SimpleOptionParser.h"
#include "SumLikelihoodPdf.h"
#include "TFile.h"
#include "TH1.h"
#include "HistogramPdf.h"
#include "GeneralConvolutionPdf.h"
#include "ResponseFunctionPdf.h"
SimpleInputBuilder::SimpleInputBuilder() :
    folder(std::getenv("SimpleInputBuilderData")?std::getenv("SimpleInputBuilderData"):""),
    spcBuilder(std::make_shared<BasicSpectrumBuilder>())  { }

std::string SimpleInputBuilder::loadOutputFileNameFromCmdArgs(int argc,char **argv) {
  if(argc<2) {
    std::cerr<<"Usage: "<<argv[0]<<" <configFile> [outputName] [key=value] [key2=value2] ..."<<std::endl;
    std::cerr<<"SimpleInputBuilder::loadConfigsFromCmdArgs aborted."<<std::endl;
    throw GooStatsException("cmd argument format not understandable");
  }
  return argc>2?std::string(argv[2]):std::string("output.root");
}

std::vector<InputConfig*> SimpleInputBuilder::loadConfigsFromCmdArgs(int argc,char **argv) {
  if(argc<2) {
    std::cerr<<"Usage: "<<argv[0]<<" <configFile> [outputName] [key=value] [key2=value2] ..."<<std::endl;
    std::cerr<<"SimpleInputBuilder::loadConfigsFromCmdArgs aborted."<<std::endl;
    throw GooStatsException("cmd argument format not understandable");
  }
  std::vector<InputConfig*> configs;
  InputConfig *config = new InputConfig;
  config->configFile = std::string(argv[1]);
  configs.push_back(config);
  return configs;
}

ConfigsetManager *SimpleInputBuilder::buildConfigset(ParSyncManager *parManager,const InputConfig &config) {
  ConfigsetManager *configset = new ConfigsetManager(*parManager->createParSyncSet(config));
  configset->setOptionManager(createOptionManager());
  return configset;
}
void SimpleInputBuilder::fillRawSpectrumProvider(RawSpectrumProvider *provider,ConfigsetManager* configset) {
  if(!configset->has("inputSpectra")) return;

  std::vector<std::string> componentsTH1;
  struct txtSource { std::string component; std::string txt; };
  std::vector<txtSource> componentsTxt;
  for(auto component : GooStats::Utility::splitter(configset->query("inputSpectra"),":"))
    if(configset->has(component + "_inputTxt")) 
      componentsTxt.push_back(
	  (txtSource) {component,configset->query(component + "_inputTxt")});
    else
      componentsTH1.push_back(component);

  // load txt
  for(auto txtPair : componentsTxt) {
    std::ifstream f;
    f.open(txtPair.txt);
    if(!f.is_open()) {
      f.open(folder + txtPair.txt);
      if(!f.is_open())
	throw GooStatsException("Cannot open <"+txtPair.txt+"> nor <"+
	    folder+">/<"+txtPair.txt+">");
    }
    int n;
    double e0,de,*x;
    f >> n >> e0 >> de;
    if(!f.good()) throw GooStatsException("Cannot read n/e0/de from <"+txtPair.txt+">");
    x = new double [n]; // provider should delete it
    for(int i = 0; i<n; ++i) {
      f>> x[i];
      if(!f.good()) {
	std::cout<<"Cannot <"<<i<<">-th pdf value from <"<<txtPair.txt<<">"<<std::endl;;
	throw GooStatsException("Cannot load pdf value from <"+txtPair.txt+">");
      }
    }
    provider->registerSpecies(txtPair.component,n,x,e0,de);
    f.close();
  }

  // load TFile
  if(componentsTH1.size() && configset->has("inputSpectraFiles")) {
    std::vector<std::string> sourceTFilesName(
	GooStats::Utility::splitter(configset->query("inputSpectraFiles"),":"));;
    std::vector<TFile *> sourceTFiles;
    for(auto fileName : sourceTFilesName) {
      TFile *file = TFile::Open(fileName.c_str());
      if(!file->IsOpen()) {
	file = TFile::Open((folder + fileName).c_str());
	if(!file->IsOpen()) {
	  throw GooStatsException("Cannot open <"+fileName+"> nor <"+
	      folder+">/<"+fileName+">");
	}
      }
      sourceTFiles.push_back(file); 
    }
    for(auto component : componentsTH1) {
      std::string histName; TH1 *th1(nullptr);
      if(configset->has(component+"_histName"))
	histName = configset->query(component+"_histName");
      for(auto file: sourceTFiles) {
	th1 = static_cast<TH1*>(file->Get(histName.c_str()));
	if(th1) break;
      }
      if(!th1) {
	std::cout<<"Cannot find <"<<histName<<"> from TFiles"<<std::endl;
	std::cout<<"List of TFiles ("<<sourceTFilesName.size()<<"): "<<std::endl;
	for(size_t i = 0;i<sourceTFilesName.size();++i) {
	  std::cout<<"["<<i<<"] <"<<sourceTFilesName.at(i)<<">"<<std::endl;
	}
	std::cout<<"List of histograms to be loaded: "<<std::endl;
	for(auto component : componentsTH1) 
	  std::cout<<"["<<component<<"] <"<<(configset->has(component+"_histName")?configset->query(component+"_histName"):"")<<">"<<std::endl;
	throw GooStatsException("Cannot load pdf from TFiles");
      }
      int n;
      double e0,de,*x;
      n = th1->GetNbinsX();
      e0 = th1->GetBinCenter(1);
      de = th1->GetBinWidth(1);
      x = new double [n]; // provider should delete it
      for(int i = 0;i<n;++i) 
	x[i] = th1->GetBinContent(i+1);
      provider->registerSpecies(component,n,x,e0,de);
    }
    for(auto file : sourceTFiles)
      file->Close(); // don't delete, ROOT will delete them 
  }
}
void SimpleInputBuilder::createVariables(ConfigsetManager* configset) {
  std::vector<std::string> components(GooStats::Utility::splitter(configset->query("components"),":"));;
  for(auto component: components) {
    // warning: no error checking
    configset->createVar(component,
	::atof(configset->query("N"+component+"_init").c_str()),
	::atof(configset->query("N"+component+"_err").c_str()),
	::atof(configset->query("N"+component+"_min").c_str()),
	::atof(configset->query("N"+component+"_max").c_str()));
  } 
}

DatasetManager *SimpleInputBuilder::buildDataset(DatasetController *controller) {
  return controller->createDataset();
}

bool SimpleInputBuilder::buildRawSpectra(DatasetManager *dataset,RawSpectrumProvider *provider) {
  spcBuilder->AddSiblings(new SimpleSpectrumBuilder(provider));
  for(auto component : dataset->get<std::vector<std::string>>("components")) {
    if((dataset->get<std::string>(component+"_type")=="Ana")||
	(dataset->get<std::string>(component+"_type")=="AnaShifted")) {
      GooPdf *innerPdf = spcBuilder->buildSpectrum(component+"_inner",dataset);
      if(!innerPdf) {
	std::cout<<"No hanlder is found to build spectrum <"<<component<<"_inner> "
	  <<"type <"<<dataset->get<std::string>(component+"_inner_type")<<">"<<std::endl;
	throw GooStatsException("Cannot build spectrum");
      }
      dataset->set<PdfBase*>(component+"_innerPdf",innerPdf);
    }
  }
  return true;
}
bool SimpleInputBuilder::buildComponenets(DatasetManager *dataset,RawSpectrumProvider *) {
  std::vector<PdfBase*> pdfs;
  for(auto component : dataset->get<std::vector<std::string>>("components")) {
    // get Raw spec
    GooPdf *pdf = spcBuilder->buildSpectrum(component,dataset);
    if(!pdf) continue; // place holder
    pdfs.push_back(pdf);
  }
  dataset->set<std::vector<PdfBase*>>("pdfs",pdfs);
  return true;
}

bool SimpleInputBuilder::fillOptions(ConfigsetManager *configset,int argc,char **argv) {
  configset->parse(argc,argv);
  return true;
}
bool SimpleInputBuilder::fillOptions(ConfigsetManager *configset,const std::string &configFile) {
  configset->parse(configFile);
  return true;
}

bool SimpleInputBuilder::installSpectrumBuilder(ISpectrumBuilder *builder) {
  spcBuilder->AddSiblings(builder);
  return true;
}

std::vector<std::shared_ptr<DatasetController>> SimpleInputBuilder::buildDatasetsControllers(ConfigsetManager *configset) {
  std::vector<std::shared_ptr<DatasetController>> controllers;
  controllers.push_back(std::shared_ptr<DatasetController>(new SimpleDatasetController(configset)));
  if(configset->has("pullPars"))
  for(auto par : GooStats::Utility::splitter(configset->query("pullPars"),":")) {
      controllers.push_back(std::shared_ptr<DatasetController>(new PullDatasetController(configset,par+"_pull")));
  }
  return controllers;
}
OptionManager *SimpleInputBuilder::createOptionManager() {
  return new SimpleOptionParser();
}
SumLikelihoodPdf *SimpleInputBuilder::buildTotalPdf(const std::vector<DatasetManager*> &datasets) {
  std::vector<PdfBase*> likelihoodTerms;
  for(auto dataset : datasets) {
    if(!dataset->getLikelihood()) continue;
    std::cout<<"Inserting <"<<dataset->name()<<">"<<std::endl;
    likelihoodTerms.push_back(static_cast<PdfBase*>(dataset->getLikelihood()));
  }
  SumLikelihoodPdf *sumpdf = new SumLikelihoodPdf("full_likelihood",likelihoodTerms);
  return sumpdf;
}
bool SimpleInputBuilder::configParameters(DatasetManager *) {
  return true;
}
