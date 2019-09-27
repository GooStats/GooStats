#include "gtest/gtest.h"
#include "BestFitFixture.h"
#include "NLLCheckFixture.h"
#include <string>
#include <iostream>
#include "GooStatsException.h"

#include "AnalysisManager.h"
#include "InputManager.h"
#include "ReactorInputBuilder.h"
#include "ReactorSpectrumBuilder.h"
#include "OutputManager.h"
#include "SimpleOutputBuilder.h"
#include "SimplePlotManager.h"
#include "GSFitManager.h"
#include "PrepareData.h"
#include "ContourManager.h"
#include "CorrelationManager.h"
#include "SimpleFit.h"
#include "ScanPar.h"
#include "NMOTest.h"

void trigger_fit(const std::string arg1 = "reactor", const std::string arg2 = "data_recE.cfg", const std::string arg3 = "NLLValid",
		 const std::string arg4 = "inputSpectraFiles=data/data_hist.root",const std::string arg5 = "main_histName=Evis_hist_poissonAppSum",
		 const std::string arg6 = "", const std::string arg7 = "", const std::string arg8 = "", const std::string arg9 = "",
		 const std::string arg10 = "", const std::string arg11 = "", const std::string arg12 = "", const std::string arg13 = "",
		 const std::string arg14 = "", const std::string arg15 = "", const std::string arg16 = "", const std::string arg17 = "",
		 const std::string arg18 = "", const std::string arg19 = "", const std::string arg20 = "", const std::string arg21 = "") {
  int argc(0);
  char **argv;
  argv = new char *[21];
  for(int i = 0;i<21;++i) argv[i] = new char[255];
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
  if(arg18!="") { std::strcpy(argv[argc],arg18.c_str()); ++argc; }
  if(arg19!="") { std::strcpy(argv[argc],arg19.c_str()); ++argc; }
  if(arg20!="") { std::strcpy(argv[argc],arg20.c_str()); ++argc; }
  if(arg21!="") { std::strcpy(argv[argc],arg21.c_str()); ++argc; }
  for(int i = 0;i<argc;++i) std::cout<<argv[i]<<" ";
  std::cout<<std::endl;

  AnalysisManager *ana = new AnalysisManager();

  InputManager *inputManager = new InputManager(argc,const_cast<const char**>(argv));
  InputBuilder *builder = new ReactorInputBuilder();
  builder->installSpectrumBuilder(new ReactorSpectrumBuilder());
  inputManager->setInputBuilder(builder);

  GSFitManager *gsFitManager = new GSFitManager();

  OutputManager *outManager = new OutputManager();
  outManager->setOutputBuilder(new SimpleOutputBuilder());

  StatModule::setup(inputManager);
  StatModule::setup(gsFitManager);
  StatModule::setup(outManager);

  PlotManager *plotManager = new SimplePlotManager();
  outManager->setPlotManager(plotManager);

  PrepareData *data = new PrepareData();
  SimpleFit *fit = new SimpleFit();
  NMOTest *nmo = new NMOTest();
  ScanPar *scan = new ScanPar();
  CorrelationManager *correlationManager = new CorrelationManager();
  ContourManager *contourManager = new ContourManager();

  ana->registerModule(inputManager);
  ana->registerModule(data);
  ana->registerModule(fit);
  ana->registerModule(nmo);
  ana->registerModule(scan);
  ana->registerModule(correlationManager);
  ana->registerModule(contourManager);
  ana->registerModule(outManager);

  ana->init();
  ana->run();
  ana->finish();
}

void load_species(std::vector<std::string> &species) {
#include "species.icc"
}

TEST_F(BestFitFixture, TAUP_npmt_near) {
  load_species(species);
  load_result("data/plotEvPPS.root",reference_fit,rough_reference_fit);
  load_result("NLLValid.root",new_fit,rough_new_fit);
  const size_t N = reference_fit.size();
  ASSERT_EQ ( 44u, N );
  const double precision = 1e-4;
  for(size_t i = 0;i<N;++i) {
    EXPECT_NEAR(reference_fit.at(i),new_fit.at(i),fabs(precision*reference_fit.at(i))) 
      << species.at(i);
  }
}
TEST_F(BestFitFixture, Asimov_near) {
  load_species(species);
  for(int sub = 0; sub<2; ++sub) {
    setEntry(0,sub);
    load_result("data/plotEvPPSHTAsimov.root",reference_fit,rough_reference_fit);
    load_result("NLLValidHTAsimov.root",new_fit,rough_new_fit);
    const size_t N = reference_fit.size();
    ASSERT_EQ ( 44u, N );
    const double precision = 1e-4;
    for(size_t i = 0;i<N;++i) {
      EXPECT_NEAR(reference_fit.at(i),new_fit.at(i),fabs(precision*reference_fit.at(i))+precision) 
        << species.at(i);
    }
  }
}
TEST_F(BestFitFixture, toyMC_near) {
  load_species(species);
  for(int ent = 0; ent<10; ++ent) {
    for(int sub = 0; sub <2; ++sub) {
      setEntry(ent,sub);
      load_result("data/plotEvPPSHTRND.root",reference_fit,rough_reference_fit);
      load_result("NLLValidHTRND.root",new_fit,rough_new_fit);
      const size_t N = reference_fit.size();
      ASSERT_EQ ( 44u, N );
      const double precision = 1e-4;
      for(size_t i = 0;i<N;++i) {
        EXPECT_NEAR(reference_fit.at(i),new_fit.at(i),fabs(precision*reference_fit.at(i))) 
          << species.at(i);
      }
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  trigger_fit("reactor","data_recE.cfg","NLLValid","inputSpectraFiles=data/data_hist.root","main_histName=Evis_hist_poissonAppSum","seed=1",
      "deltaM231_err=0.18","deltaM221_err=0.18",
      "print_contour=true","plot_profiles=default.deltaM231","plot_contours=default.Reactor:default.deltaM231;default.Reactor:default.deltaM221",
      "contour_N=4",
      "corr_variables=default.Reactor:default.deltaM231:default.deltaM221",
      "label_default.Reactor=N_{REA}","label_default.deltaM231=#deltam_{32}^{2}","label_default.deltaM221=#deltam_{21}^{2}",
      "pullPars=Reactor:deltaM231","Reactor_pullType=square","Reactor_min=0.9","Reactor_max=1");
  trigger_fit("reactor","data_recE.cfg","NLLValidHTAsimov","inputSpectraFiles=data/data_hist.root","main_histName=Evis_hist_poissonAppSum","seed=1",
      "fitFakeData=true","fitAsimov=true","fitNMO=true","SimpleFit=false");
  trigger_fit("reactor","data_recE.cfg","NLLValidHTRND","inputSpectraFiles=data/data_hist.root","main_histName=Evis_hist_poissonAppSum","seed=1",
      "fitFakeData=true","repeat=10","seed=1","fitNMO=true","SimpleFit=false");
  return RUN_ALL_TESTS();
}
