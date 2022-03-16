#include <unistd.h>

#include "GooStatsNLLCheck.h"
#include "TFile.h"
#include "fit.h"
#include "gtest/gtest.h"

TEST(GooStats, NLLTest) {
  try {
    const char *argv[] = {"GooStats.exe", "toyMC.cfg", "test", "dummyOption=ShouldDiscardIt"};
    GooStats::fit(sizeof argv / sizeof argv[0], argv);
  } catch (std::exception &ex) {
    std::cerr << "Caught [" << ex.what() << "]" << std::endl;
  }

  char cwd[255];
  if (getcwd(cwd, sizeof(cwd)) != NULL) {
    printf("Current working dir: %s\n", cwd);
  }
  TFile *results_f = TFile::Open("NLL_CHECK_gpu.root");
  auto results_obj = static_cast<GooStatsNLLCheck *>(results_f->Get("gpu"));
  auto results = (results_obj->get_results());
  TFile *reference_f = TFile::Open("NLL_CHECK_reference.root");
  auto reference_obj = static_cast<GooStatsNLLCheck *>(reference_f->Get("gpu"));
  auto references = (reference_obj->get_results());
  for (size_t i = 0; i < results.size(); ++i) {
    for (auto x : results[i]) {
      EXPECT_NEAR(x.second.LL, references[i][x.first].LL, x.second.LL * 5e-11);
    }
  }
  EXPECT_NEAR(results_obj->get_finalLL(), reference_obj->get_finalLL(), reference_obj->get_finalLL() * 5e-11);
}

TEST(GooStats, NLLTestFail) {
  try {
    const char *argv[] = {"GooStats.exe", "toyMC.cfg", "test", "dummyOption=ShouldDiscardIt"};
    GooStats::fit(sizeof argv / sizeof argv[0], argv);
  } catch (std::exception &ex) {
    std::cerr << "Caught [" << ex.what() << "]" << std::endl;
  }

  char cwd[255];
  if (getcwd(cwd, sizeof(cwd)) != NULL) {
    printf("Current working dir: %s\n", cwd);
  }
  TFile *results_f = TFile::Open("NLL_CHECK_gpu.root");
  auto results_obj = static_cast<GooStatsNLLCheck *>(results_f->Get("gpu"));
  auto results = (results_obj->get_results());
  TFile *reference_f = TFile::Open("NLL_CHECK_reference.root");
  auto reference_obj = static_cast<GooStatsNLLCheck *>(reference_f->Get("gpu"));
  auto references = (reference_obj->get_results());
  EXPECT_NEAR(
      results_obj->get_finalLL() * 100 + 100, reference_obj->get_finalLL(), reference_obj->get_finalLL() * 5e-11);
}
