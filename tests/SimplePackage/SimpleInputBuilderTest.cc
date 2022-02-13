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

TEST(GooStats, SimpleInputBuilder) {
  InputBuilder *builder = new SimpleInputBuilder();
  int argc = 5;
  const char *argv[] = {"run.exe","test.cfg","output","arg1=33","arg2=99a"};
  auto output = builder->loadOutputFileName(argc, argv);
  ASSERT_EQ(output,"output");
  ASSERT_NE(output,"something else");

  ParSyncManager parSyncManager;
  auto configs = builder->buildConfigsetManagers(&parSyncManager,argc,argv);
  ASSERT_EQ(configs.size(),1);
  auto config = configs.at(0);
  ASSERT_EQ(config->get("pullPars"),"A:B:C.D.E : F :G");
  auto controllers = builder->buildDatasetsControllers(config);
  ASSERT_EQ(controllers.size(),6);
  ASSERT_NE(dynamic_cast<SimpleDatasetController*>(controllers.at(0).get()),nullptr);
  ASSERT_NE(dynamic_cast<PullDatasetController*>(controllers.at(5).get()),nullptr);
}