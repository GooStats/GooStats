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
#include "GSFitManager.h"
#include "ContourManager.h"
#include "CorrelationManager.h"

void trigger_fit(const std::string arg1 = "reactor", const std::string arg2 = "data_recE.cfg", const std::string arg3 = "NLLValid",
		 const std::string arg4 = "inputSpectraFiles=data/data_hist.root",const std::string arg5 = "Data_histName=Evis_hist_poissonAppSum",
		 const std::string arg6 = "", const std::string arg7 = "", const std::string arg8 = "", const std::string arg9 = "",
		 const std::string arg10 = "", const std::string arg11 = "", const std::string arg12 = "", const std::string arg13 = "",
		 const std::string arg14 = "", const std::string arg15 = "", const std::string arg16 = "", const std::string arg17 = "") {
  int argc(0);
  char **argv;
  argv = new char *[17];
  for(int i = 0;i<17;++i) argv[i] = new char[255];
  if(arg1!="") { std::strcpy(argv[argc],arg1.c_str()); ++argc; }
  if(arg2!="") { std::strcpy(argv[argc],arg2.c_str()); ++argc; }
  if(arg3!="") { std::strcpy(argv[argc],arg3.c_str()); ++argc; }
  if(arg4!="") { std::strcpy(argv[argc],arg4.c_str()); ++argc; }
  if(arg5!="") { std::strcpy(argv[argc],arg5.c_str()); ++argc; }
  if(arg6!="") { std::strcpy(argv[argc],arg6.c_str()); ++argc; }
  if(arg7!="") { std::strcpy(argv[argc],arg7.c_str()); ++argc; }
  if(arg8!="") { std::strcpy(argv[argc],arg8.c_str()); ++argc; }
  if(arg9!="") { std::strcpy(argv[argc],arg9.c_str()); ++argc; }
  if(arg10!="") { std::strcpy(argv[argc],arg10.c_str()); ++argc; }
  if(arg11!="") { std::strcpy(argv[argc],arg11.c_str()); ++argc; }
  if(arg12!="") { std::strcpy(argv[argc],arg12.c_str()); ++argc; }
  if(arg13!="") { std::strcpy(argv[argc],arg13.c_str()); ++argc; }
  if(arg14!="") { std::strcpy(argv[argc],arg14.c_str()); ++argc; }
  if(arg15!="") { std::strcpy(argv[argc],arg15.c_str()); ++argc; }
  if(arg16!="") { std::strcpy(argv[argc],arg16.c_str()); ++argc; }
  if(arg17!="") { std::strcpy(argv[argc],arg17.c_str()); ++argc; }
  for(int i = 0;i<argc;++i) std::cout<<argv[i]<<" ";
  std::cout<<std::endl;

  AnalysisManager *ana = new ReactorAnalysisManager();
  InputManager *inputManager = new InputManager(argc,argv);
  InputBuilder *builder = new ReactorInputBuilder();
  builder->installSpectrumBuilder(new ReactorSpectrumBuilder());
  inputManager->setInputBuilder(builder);
  ana->setInputManager(inputManager);
  ana->registerModule(inputManager);

  GSFitManager *gsFitManager = new GSFitManager();
  gsFitManager->registerDependence(inputManager);
  ana->registerModule(gsFitManager);

  OutputManager *outManager = new OutputManager();
  outManager->registerDependence(inputManager);
  outManager->registerDependence(gsFitManager);
  outManager->setOutputBuilder(new SimpleOutputBuilder());
  ana->setOutputManager(outManager);
  ana->registerModule(outManager);

  PlotManager *plotManager = new SimplePlotManager();
  plotManager->registerDependence(gsFitManager);
  plotManager->registerDependence(outManager);
  plotManager->registerDependence(inputManager);
  outManager->setPlotManager(plotManager);


  CorrelationManager *correlationManager = new CorrelationManager();
  correlationManager->registerDependence(inputManager);
  correlationManager->registerDependence(gsFitManager);
  correlationManager->registerDependence(outManager);
  ana->registerModule(correlationManager);

  ContourManager *contourManager = new ContourManager();
  contourManager->registerDependence(inputManager);
  contourManager->registerDependence(gsFitManager);
  contourManager->registerDependence(outManager);
  ana->registerModule(contourManager);


  ana->init();
  ana->run();
  ana->finish();
}

void load_species(std::vector<std::string> &species) {
#include "species.icc"
}

TEST_F(BestFitFixture, TAUP_npmt_exact) {
  load_species(species);
  load_result("data/plotEvPPS.root",reference_fit,rough_reference_fit);
  load_result("NLLValid.root",new_fit,rough_new_fit);
  const size_t N = reference_fit.size();
  ASSERT_EQ ( 44u, N );
  for(size_t i = 0;i<N;++i) {
    EXPECT_DOUBLE_EQ(reference_fit.at(i),new_fit.at(i))
      << species.at(i);
  }
}
TEST_F(BestFitFixture, TAUP_npmt_near) {
  load_species(species);
  load_result("data/plotEvPPS.root",reference_fit,rough_reference_fit);
  load_result("NLLValid.root",new_fit,rough_new_fit);
  const size_t N = reference_fit.size();
  ASSERT_EQ ( 44u, N );
  const double precision = 0.001;
  for(size_t i = 0;i<N;++i) {
    EXPECT_NEAR(reference_fit.at(i),new_fit.at(i),fabs(precision*reference_fit.at(i))) 
      << species.at(i);
  }
}
TEST_F(BestFitFixture, Asimov_exact) {
  load_species(species);
  for(int subEntry = 0; subEntry<2; ++subEntry) {
    setEntry(0,subEntry);
    load_result("data/plotEvPPSHTAsimov.root",reference_fit,rough_reference_fit);
    load_result("NLLValidHTAsimov.root",new_fit,rough_new_fit);
    const size_t N = reference_fit.size();
    ASSERT_EQ ( 44u, N );
    for(size_t i = 0;i<N;++i) {
      EXPECT_DOUBLE_EQ(reference_fit.at(i),new_fit.at(i))
	<< species.at(i);
    }
  }
}
TEST_F(BestFitFixture, toyMC_exact) {
  load_species(species);
  for(int entry = 0; entry<10; ++entry) {
    for(int subEntry = 0; subEntry<2; ++subEntry) {
      setEntry(entry,subEntry);
      load_result("data/plotEvPPSHTRND.root",reference_fit,rough_reference_fit);
      load_result("NLLValidHTRND.root",new_fit,rough_new_fit);
      const size_t N = reference_fit.size();
      ASSERT_EQ ( 44u, N );
      for(size_t i = 0;i<N;++i) {
	EXPECT_DOUBLE_EQ(reference_fit.at(i),new_fit.at(i))
	  << species.at(i);
      }
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  trigger_fit("reactor","data_recE.cfg","NLLValid","inputSpectraFiles=data/data_hist.root","Data_histName=Evis_hist_poissonAppSum","seed=1",
	      "deltaM231_err=0.18","deltaM221_err=0.18",
	      "print_contour=true","plot_profiles=default.NReactor","plot_contours=default.NReactor:default.deltaM231;default.NReactor:default.deltaM221",
	      "corr_variables=default.NReactor:default.deltaM231:default.deltaM221",
	      "label_default.NReactor=N_{REA}","label_default.deltaM231=#deltam_{32}^{2}","label_default.deltaM221=#deltam_{21}^{2}");
  trigger_fit("reactor","data_recE.cfg","NLLValidHTAsimov","inputSpectraFiles=data/data_hist.root","Data_histName=Evis_hist_poissonAppSum","seed=1",
	      "fitFakeData=true","fitAsimov=true","fitInverseMH=true");
  trigger_fit("reactor","data_recE.cfg","NLLValidHTRND","inputSpectraFiles=data/data_hist.root","Data_histName=Evis_hist_poissonAppSum","seed=1",
	      "fitFakeData=true","repeat=10","seed=1","fitInverseMH=true");
  return RUN_ALL_TESTS();
}
