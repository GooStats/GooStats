/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "OutputHelper.h"
#include "fit.h"
#include "gtest/gtest.h"

TEST(GooStats, simpleFit) {
  const char *argv[] = {"GooStats.exe", "toyMC.cfg", "test", "dummyOption=ShouldDiscardIt"};
  auto outputHelper = GooStats::fit(sizeof argv / sizeof argv[0], argv);

  EXPECT_NEAR(outputHelper->value("likelihood"), 357.95898, 0.00001);

  auto lsan = std::getenv("LSAN_OPTIONS");
  std::cout<<"LSAN_OPTIONS: ["<<(lsan?lsan:"")<<"]"<<std::endl;
}
