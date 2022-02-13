/*****************************************************************************/
// Author: Xuefeng Ding <xuefengd@princeton.edu>
// Institute: Department of Physics, Princeton University
// Date: 2/13/22
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2022 copyrighted.
/*****************************************************************************/
#include "gtest/gtest.h"
#include "Utility.h"

TEST(GooStats, UtilityTest) {
  auto splitted = GooStats::Utility::splitter(" A : Bfa : fs:d  ",":");
  ASSERT_EQ(splitted.size(),4);
  ASSERT_EQ(splitted.at(0),"A");
  ASSERT_EQ(splitted.at(3),"d");
}