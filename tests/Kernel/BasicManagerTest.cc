/*****************************************************************************/
// Author: Xuefeng Ding <xuefengd@princeton.edu>
// Institute: Department of Physics, Princeton University
// Date: 2/12/22
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2022 copyrighted.
/*****************************************************************************/
#include "BasicManager.h"
#include "goofit/Variable.h"
#include "gtest/gtest.h"

TEST(GooStats, BasicManagerTest) {
  auto manager = std::make_shared<BasicManager>("PhaseI.TFCsub.MLPsub");
  auto manager2 = std::make_shared<BasicManager>("PhaseI.TFCcomp.MLPsub");
  auto manager3 = std::make_shared<BasicManager>("PhaseII");
  BasicManager::setParSyncConfig({{"global_var", 0}, {"phase", 1}, {"TFC", 2}, {"MLP", 3}, {"other", 99}});

  Variable *global = manager->createVar("global_var", 1, 2, 3, 4);
  ASSERT_EQ(global->name, "global.global_var");
  ASSERT_NE(global->name, "something different");
  Variable *global2 = manager2->createVar("global_var", 6, 5, 4, 3);
  ASSERT_EQ(global, global2);
  ASSERT_EQ(global2->lowerlimit, 3);

  Variable *phase1 = manager->createVar("phase", 1, 2, 3, 4);
  Variable *phase1s = manager2->createVar("phase", 5, 6, 7, 8);
  Variable *phase2 = manager3->createVar("phase", 9, 10, 11, 12);
  ASSERT_EQ(phase1, phase1s);
  ASSERT_NE(phase1, phase2);

  Variable *tfc1 = manager->createVar("TFC", 1, 2, 3, 4);
  Variable *tfc2 = manager2->createVar("TFC", 5, 6, 7, 8);
  Variable *tfc3 = manager3->createVar("TFC", 9, 10, 11, 12);
  ASSERT_NE(tfc1, tfc2);
  ASSERT_NE(tfc1, tfc3);
  ASSERT_NE(tfc2, tfc3);

  Variable *mlp1 = manager->createVar("MLP", 1, 2, 3, 4);
  Variable *mlp2 = manager2->createVar("MLP", 5, 6, 7, 8);
  Variable *mlp3 = manager3->createVar("MLP", 9, 10, 11, 12);
  ASSERT_NE(mlp1, mlp2);
  ASSERT_NE(mlp1, mlp3);
  ASSERT_NE(mlp2, mlp3);

  Variable *other1 = manager->createVar("other", 1, 2, 3, 4);
  Variable *other2 = manager2->createVar("other", 5, 6, 7, 8);
  Variable *other3 = manager3->createVar("other", 9, 10, 11, 12);
  ASSERT_NE(other1, other2);
  ASSERT_NE(other1, other3);
  ASSERT_NE(other2, other3);

  Variable *extra1 = manager->createVar("extra", 1, 2, 3, 4);
  Variable *extra2 = manager2->createVar("extra", 5, 6, 7, 8);
  Variable *extra3 = manager3->createVar("extra", 9, 10, 11, 12);
  ASSERT_NE(extra1, extra2);
  ASSERT_NE(extra1, extra3);
  ASSERT_NE(extra2, extra3);

  Variable *extra4 = manager2->linkVar("extra2", "extra");
  ASSERT_EQ(extra4, extra2);
}
