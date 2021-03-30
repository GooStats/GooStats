/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "gtest/gtest.h"
#include "fit.h"
#include "OutputHelper.h"

TEST (GooStats, simpleFit) {
  auto outputHelper = GooStats::fit();

  EXPECT_NEAR(outputHelper->value("likelihood"),357.95898,0.00001);
}
