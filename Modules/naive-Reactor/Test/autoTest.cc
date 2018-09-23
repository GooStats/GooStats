#include "gtest/gtest.h"
#include "BestFitFixture.h"
#include "NLLCheckFixture.h"
#include <string>
#include <iostream>
#include "GooStatsException.h"

#include "ReactorAnalysisManager.h"
#include "InputManager.h"
#include "ReactorInputBuilder.h"
#include "ReactorSpectrumBuilder.h"
#include "OutputManager.h"
#include "SimpleOutputBuilder.h"
#include "SimplePlotManager.h"

void trigger_fit(const std::string &exe,const std::string &cfg,const std::string &fname,const std::string &hname) {
  int argc(5);
  char **argv;
  argv = new char *[5];
  for(int i = 0;i<5;++i) argv[i] = new char[255];
  std::strcpy(argv[0],exe.c_str());
  std::strcpy(argv[1],cfg.c_str());
  std::strcpy(argv[2],"NLLValid");
  std::strcpy(argv[3],("inputSpectraFiles="+fname).c_str());
  std::strcpy(argv[4],("Data_histName="+hname).c_str());
  for(int i = 0;i<5;++i) std::cout<<argv[i]<<" ";
  std::cout<<std::endl;

  AnalysisManager *ana = new ReactorAnalysisManager();
  InputManager *inputManager = new InputManager(argc,argv);
  InputBuilder *builder = new ReactorInputBuilder();
  builder->installSpectrumBuilder(new ReactorSpectrumBuilder());
  inputManager->setInputBuilder(builder);
  ana->setInputManager(inputManager);
  OutputManager *outManager = new OutputManager();
  outManager->setOutputBuilder(new SimpleOutputBuilder());
  outManager->setPlotManager(new SimplePlotManager());
  ana->setOutputManager(outManager);

  ana->init();
  ana->run();
  ana->finish();
}

void load_species(std::vector<std::string> &species) {
#include "species.icc"
}

TEST_F(BestFitFixture, TAUP_npmt_exact) {
  load_species(species);
  load_result("data/plotEvPPS_tree.root",reference_fit,rough_reference_fit);
  load_result("NLLValid_tree.root",new_fit,rough_new_fit);
  const size_t N = reference_fit.size();
  ASSERT_EQ ( 42u, N );
  for(size_t i = 0;i<N;++i) {
    EXPECT_DOUBLE_EQ(reference_fit.at(i),new_fit.at(i))
      << species.at(i);
  }
}
TEST_F(BestFitFixture, TAUP_npmt_near) {
  load_species(species);
  load_result("data/plotEvPPS_tree.root",reference_fit,rough_reference_fit);
  load_result("NLLValid_tree.root",new_fit,rough_new_fit);
  const size_t N = reference_fit.size();
  ASSERT_EQ ( 42u, N );
  const double precision = 0.001;
  for(size_t i = 0;i<N;++i) {
    EXPECT_NEAR(reference_fit.at(i),new_fit.at(i),fabs(precision*reference_fit.at(i))) 
      << species.at(i);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  trigger_fit("reactor","data_recE.cfg","data/data_hist.root","Evis_hist_poissonAppSum");
  return RUN_ALL_TESTS();
}
