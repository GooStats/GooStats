#include "gtest/gtest.h"
#include "GooStatsNLLCheck.h"
#include "TFile.h"
#include <unistd.h>


TEST (GooFit, NLLTest) {
  TFile *results_f = TFile::Open("NLL_CHECK_gpu.root");
  auto results_obj = static_cast<GooStatsNLLCheck*>(results_f->Get("gpu"));
  auto results = (results_obj->get_results());
  TFile *reference_f = TFile::Open("NLL_CHECK_reference.root");
  auto reference_obj = static_cast<GooStatsNLLCheck*>(reference_f->Get("gpu"));
  auto references = (reference_obj->get_results());
  for(size_t i = 0;i<results.size();++i) {
    for(auto x: results[i]) {
      EXPECT_NEAR(x.second.LL, references[i][x.first].LL,x.second.LL*5e-11);
    }
  }
}
