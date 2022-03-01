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
#include "ConfigsetManager.h"
#include "DatasetController.h"
#include "DatasetManager.h"
#include "GooStatsException.h"
#include "HistogramPdf.h"
#include "ParSyncManager.h"
#include "PullDatasetController.h"
#include "RawSpectrumProvider.h"
#include "SimpleDatasetController.h"
#include "SimpleOptionParser.h"
#include "SimpleSpectrumBuilder.h"
#include "SumLikelihoodPdf.h"
#include "TFile.h"
#include "TH1.h"
#include "Utility.h"
#include "goofit/Variable.h"
#include <fstream>
#include <iostream>


std::string SimpleInputBuilder::loadOutputFileName(int argc, const char **argv,
                                                   std::vector<ConfigsetManager *> configsets) {
  if (argc < 2) {
    usage(argv);
    throw GooStatsException("cmd argument format not understandable");
  }
  if (argc == 2) {
    for (auto config: configsets) {
      if (config->has("output")) { return config->get("output"); }
    }
    std::cerr << "Warning: output not set, <output> is used. Set output name as 3rd argument or set [output=."
                 "..] in your config files."
              << std::endl;
    return "output";
  } else {
    return argv[2];
  }
}
void SimpleInputBuilder::usage(const char *const *argv) const {
  std::cerr << "Usage: " << argv[0] << " <configFile> [outputName] [key=value] [key2=value2] ..." << std::endl;
}

std::pair<ConfigsetManager *, std::vector<ConfigsetManager *>>
SimpleInputBuilder::buildConfigsetManagers(ParSyncManager *parManager, int argc,
                                                                           const char **argv) {
  if (argc < 2) {
    usage(argv);
    throw GooStatsException("cmd argument format not understandable");
  }
  std::vector<ConfigsetManager *> configs;
  auto configset = new ConfigsetManager("default", new OptionManager());
  auto parser = new SimpleOptionParser();
  parser->parse(configset, argv[1]);
  parser->parse(configset, argc - 3, argv + 3);
  configs.push_back(configset);
  return std::make_pair(configs.at(0),configs);
}

void SimpleInputBuilder::fillRawSpectrumProvider(RawSpectrumProvider *provider, ConfigsetManager *configset) {
  if (!configset->has("inputSpectra")) return;
  std::string folder{std::getenv("SimpleInputBuilderData") ? std::getenv("SimpleInputBuilderData") : ""};
  std::vector<std::string> componentsTH1;
  struct txtSource {
    std::string component;
    std::string txt;
  };
  std::vector<txtSource> componentsTxt;
  for (const auto &component: GooStats::Utility::split(configset->get("inputSpectra"), ":"))
    if (configset->has(component + "_inputTxt"))
      componentsTxt.push_back({component, configset->get(component + "_inputTxt")});
    else
      componentsTH1.push_back(component);

  // load txt
  for (const auto &txtPair: componentsTxt) {
    std::ifstream f;
    f.open(txtPair.txt);
    if (!f.is_open()) {
      f.open(folder + txtPair.txt);
      if (!f.is_open())
        throw GooStatsException("Cannot open <" + txtPair.txt + "> nor <" + folder + ">/<" + txtPair.txt + ">");
    }
    int n;
    double e0, de, *x;
    f >> n >> e0 >> de;
    if (!f.good()) throw GooStatsException("Cannot read n/e0/de from <" + txtPair.txt + ">");
    x = new double[n];// provider should delete it
    for (int i = 0; i < n; ++i) {
      f >> x[i];
      if (!f.good()) {
        std::cout << "Cannot <" << i << ">-th pdf value from <" << txtPair.txt << ">" << std::endl;
        ;
        throw GooStatsException("Cannot load pdf value from <" + txtPair.txt + ">");
      }
    }
    provider->registerSpecies(txtPair.component, n, x, e0, de);
    f.close();
  }

  // load TFile
  if (!componentsTH1.empty() && configset->has("inputSpectraFiles")) {
    std::vector<std::string> sourceTFilesName(GooStats::Utility::split(configset->get("inputSpectraFiles"), ":"));
    ;
    std::vector<TFile *> sourceTFiles;
    for (const auto &fileName: sourceTFilesName) {
      TFile *file = TFile::Open(fileName.c_str());
      if (!file->IsOpen()) {
        file = TFile::Open((folder + fileName).c_str());
        if (!file->IsOpen()) {
          throw GooStatsException("Cannot open <" + fileName + "> nor <" + folder + ">/<" + fileName + ">");
        }
      }
      sourceTFiles.push_back(file);
    }
    for (auto component: componentsTH1) {
      std::string histName;
      TH1 *th1(nullptr);
      if (configset->has(component + "_histName")) histName = configset->get(component + "_histName");
      for (auto file: sourceTFiles) {
        th1 = static_cast<TH1 *>(file->Get(histName.c_str()));
        if (th1) break;
      }
      if (!th1) {
        std::cout << "Cannot find <" << histName << "> from TFiles" << std::endl;
        std::cout << "List of TFiles (" << sourceTFilesName.size() << "): " << std::endl;
        for (size_t i = 0; i < sourceTFilesName.size(); ++i) {
          std::cout << "[" << i << "] <" << sourceTFilesName.at(i) << ">" << std::endl;
        }
        std::cout << "List of histograms to be loaded: " << std::endl;
        for (auto component: componentsTH1)
          std::cout << "[" << component << "] <"
                    << (configset->has(component + "_histName") ? configset->get(component + "_histName") : "") << ">"
                    << std::endl;
        throw GooStatsException("Cannot load pdf from TFiles");
      }
      int n;
      double e0, de, *x;
      n = th1->GetNbinsX();
      e0 = th1->GetBinCenter(1);
      de = th1->GetBinWidth(1);
      x = new double[n];// provider should delete it
      for (int i = 0; i < n; ++i) x[i] = th1->GetBinContent(i + 1);
      provider->registerSpecies(component, n, x, e0, de);
    }
    for (auto file: sourceTFiles) file->Close();// don't delete, ROOT will delete them
  }
}

void SimpleInputBuilder::createVariables(ConfigsetManager *configset) {
  if (configset->has("components")) {
    std::vector<std::string> pars(GooStats::Utility::split(configset->get("components"), ":"));
    for (const auto &par: pars) {
      // warning: no error checking
      auto var =
              configset->createVar(par, configset->getOrConvert(par + "_init"), configset->getOrConvert(par + "_err"),
                                   configset->getOrConvert(par + "_min"), configset->getOrConvert(par + "_max"));
      if (configset->hasAndYes(par + "_fixed")) var->fixed = true;
    }
  }
  if (configset->has("nuisance")) {
    std::vector<std::string> pars(GooStats::Utility::split(configset->get("nuisance"), ":"));
    for (const auto &par: pars) {
      auto var =
              configset->createVar(par, configset->getOrConvert(par + "_init"), configset->getOrConvert(par + "_err"),
                                   configset->getOrConvert(par + "_min"), configset->getOrConvert(par + "_max"));
      if (configset->hasAndYes(par + "_fixed")) var->fixed = true;
    }
  }
}
#include "goofit/Variable.h"
PdfBase *SimpleInputBuilder::recursiveBuild(const std::string &name, DatasetManager *dataset,
                                            RawSpectrumProvider *provider, ISpectrumBuilder *spcBuilder) {
  auto type = dataset->get<std::string>(name + "_type");
  if (dataset->has<std::string>(name + ".deps")) {
    for (auto comp: GooStats::Utility::split(dataset->get<std::string>(name + ".deps"), ":")) {
      auto innerPdf = this->recursiveBuild(comp, dataset, provider, spcBuilder);
      if (!innerPdf) {
        std::cout << "Failed to build the dependence of <" << comp << "> "
                  << "type <" << dataset->get<std::string>(comp + "_type") << ">" << std::endl;
        throw GooStatsException("Cannot build spectrum");
      }
      dataset->set<PdfBase *>(name, innerPdf);
    }
  }
  auto pdf = spcBuilder->buildSpectrum(name, dataset);
  if (!pdf) throw GooStatsException("Empty pdf");
  dataset->set(name, static_cast<PdfBase *>(pdf));
  return dataset->get<PdfBase *>(name);
}
bool SimpleInputBuilder::buildComponenets(DatasetManager *dataset, RawSpectrumProvider *provider,
                                          ISpectrumBuilder *spcBuilder) {
  std::vector<PdfBase *> pdfs;
  for (const auto &component: dataset->get<std::vector<std::string>>("components")) {
    pdfs.push_back(recursiveBuild(component, dataset, provider, spcBuilder));
  }
  dataset->set<std::vector<PdfBase *>>("pdfs", pdfs);
  return true;
}

bool SimpleInputBuilder::installSpectrumBuilder(ISpectrumBuilder *builder, RawSpectrumProvider *provider) {
  builder->AddSiblings(new SimpleSpectrumBuilder(provider));
  return true;
}

std::vector<std::shared_ptr<DatasetController>>
SimpleInputBuilder::buildDatasetsControllers(ConfigsetManager *configset) {
  std::vector<std::shared_ptr<DatasetController>> controllers;
  controllers.push_back(std::shared_ptr<DatasetController>(new SimpleDatasetController(configset)));
  if (configset->has("pullPars"))
    for (const auto &par: GooStats::Utility::split(configset->get("pullPars"), ":")) {
      controllers.push_back(std::shared_ptr<DatasetController>(new PullDatasetController(configset, par + "_pull")));
    }
  return controllers;
}
SumLikelihoodPdf *SimpleInputBuilder::buildTotalPdf(const std::vector<DatasetManager *> &datasets) {
  std::vector<PdfBase *> likelihoodTerms;
  for (auto dataset: datasets) {
    if (!dataset->getLikelihood()) {
      std::cerr << "Likelihood of dataset <" << dataset->fullName() << "> is empty." << std::endl;
      throw GooStatsException("Empty likelihood found");
    }
    std::cout << "Inserting (Datasets:" << dataset->fullName() << ")<" << dataset->getLikelihood()->getName() << ">"
              << std::endl;
    likelihoodTerms.push_back(static_cast<PdfBase *>(dataset->getLikelihood()));
  }
  std::cout << "Debug: LL size " << likelihoodTerms.size() << std::endl;

  auto sumpdf = new SumLikelihoodPdf("full_likelihood", likelihoodTerms);
  return sumpdf;
}
bool SimpleInputBuilder::fillDataSpectra(DatasetManager *dataset, RawSpectrumProvider *provider) {
  Variable *Evis = dataset->get<Variable *>("Evis");
  auto binned_data = new BinnedDataSet(Evis, dataset->fullName() + "_" + Evis->name);
  for (int i = 0; i < Evis->numbins; ++i) binned_data->setBinContent(i, provider->pdf(dataset->fullName())[i]);
  dataset->set("data", binned_data);
  return true;
}
