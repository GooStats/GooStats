/*****************************************************************************/
// Author: Xuefeng Ding <xuefengd@princeton.edu>
// Institute: Department of Physics, Princeton University
// Date: 2/13/22
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2022 copyrighted.
/*****************************************************************************/
#include "SimpleInputBuilder.h"
#include "ConfigsetManager.h"
#include "gtest/gtest.h"
#include "ParSyncManager.h"
#include "SimpleDatasetController.h"
#include "Utility.h"
#include "PullDatasetController.h"
#include "RawSpectrumProvider.h"
#include "BasicSpectrumBuilder.h"
#include "goofit/BinnedDataSet.h"
#include "goofit/PDFs/GooPdf.h"

TEST(GooStats, SimpleInputBuilder) {
  int argc = 5;
  const char *argv[] = {"run.exe","../toyMC.cfg","output","arg1=33","arg2=99a"};
  InputBuilder *builder = new SimpleInputBuilder();

  /// initializeConfigset
  ParSyncManager parSyncManager;
  auto configs_pair  = builder->buildConfigsetManagers(&parSyncManager,argc,argv);
  auto configs = configs_pair.second;
  ASSERT_EQ(configs.size(),1);
  auto config = configs.at(0);
  ASSERT_EQ(config->get("pullPars"),"gaus_Epeak");

  builder->createVariables(config);
  ASSERT_DOUBLE_EQ(config->var("gaus")->value,10);
  BasicManager::dump();

  /// installSpectrumBuilder
  BasicSpectrumBuilder spcBuilder;
  RawSpectrumProvider provider;
  builder->installSpectrumBuilder(&spcBuilder,&provider);

  /// load outName
  auto output = builder->loadOutputFileName(argc, argv);
  ASSERT_EQ(output,"output");
  ASSERT_NE(output,"something else");

  /// fillRawSpectrumProvider
  builder->fillRawSpectrumProvider(&provider, config);
  ASSERT_EQ(provider.n("default.main"),100);
  ASSERT_EQ(provider.n("fbkg"),100);

  /// initializeDatasets
  auto controllers = builder->buildDatasetsControllers(config);
  ASSERT_EQ(controllers.size(),2);
  auto major = controllers.at(0);
  ASSERT_NE(dynamic_cast<SimpleDatasetController*>(major.get()),nullptr);
  auto majorData = major->createDataset();
  ASSERT_NE(majorData,nullptr);

  major->collectInputs(majorData);
  ASSERT_EQ(majorData->get<std::vector<std::string>>("components"), std::vector<std::string> ({"gaus","fbkg"}));

  builder->fillDataSpectra(majorData,&provider);
  auto data = majorData->get<BinnedDataSet*>("data");
  ASSERT_EQ(data->getBinContent(20),226);

  builder->buildComponenets(majorData, &provider, &spcBuilder);
  auto pdfs = majorData->get<std::vector<PdfBase*>>("pdfs");
  auto gaus = static_cast<GooPdf*>(pdfs.at(0));
  auto eVis = majorData->get<Variable*>("Evis");
  {
    gaus->setData(data);
    std::vector<double> points;
    gaus->evaluateAtPoints(eVis, points);
    ASSERT_DOUBLE_EQ(points.at(50), 0.0020345763350996088);
  }

  major->buildLikelihood(majorData);
  auto majorSumPdf = majorData->getLikelihood();
  {
    majorSumPdf->setData(data);
    std::vector<double> points;
    majorSumPdf->evaluateAtPoints(eVis, points);
    ASSERT_DOUBLE_EQ(points.at(50), 120.34576335099609);
  }

  auto pull = controllers.at(1);
  ASSERT_NE(dynamic_cast<PullDatasetController*>(pull.get()),nullptr);

}