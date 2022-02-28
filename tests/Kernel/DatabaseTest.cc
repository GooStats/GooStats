/*****************************************************************************/
// Author: Xuefeng Ding <xuefengd@princeton.edu>
// Institute: Department of Physics, Princeton University
// Date: 2/28/22
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2022 copyrighted.
/*****************************************************************************/
#include "Database.h"
#include "gtest/gtest.h"

TEST(GooStats, DatabaseTest) {
  Database a;
  a.set("A", std::string("99"));
  ASSERT_EQ(a.get("A"), "99");
  std::map<std::string,std::string> ref {{"A","99"}};
  ASSERT_EQ(const_cast<const Database&>(a).list(), ref);
}