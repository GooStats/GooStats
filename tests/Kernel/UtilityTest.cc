/*****************************************************************************/
// Author: Xuefeng Ding <xuefengd@princeton.edu>
// Institute: Department of Physics, Princeton University
// Date: 2/13/22
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2022 copyrighted.
/*****************************************************************************/
#include "Utility.h"
#include "gtest/gtest.h"

TEST(GooStats, UtilityTest) {
  auto splitted = GooStats::Utility::split(" A : Bfa : fs:d  ", ":");
  ASSERT_EQ(splitted.size(), 4);
  ASSERT_EQ(splitted.at(0), "A");
  ASSERT_EQ(splitted.at(3), "d");

  auto strip = GooStats::Utility::strip(" //A : Bfa : fs:d  ");
  ASSERT_EQ(strip, "");

  auto strip2 = GooStats::Utility::strip(" B: fdsa: //A : Bfa : fs:d  ");
  ASSERT_EQ(strip2, "B: fdsa:");

  auto splitted2 = GooStats::Utility::split(strip2, ":");
  ASSERT_EQ(splitted2.size(), 2);
  ASSERT_EQ(splitted2.at(0), "B");
  ASSERT_EQ(splitted2.at(1), "fdsa");
}