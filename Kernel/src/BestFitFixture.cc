#include "BestFitFixture.h"
#include "GooStatsException.h"
#include "TFile.h"
#include "TTree.h"
#include "TTreeFormula.h"

BestFitFixture::BestFitFixture() {
}
BestFitFixture::~BestFitFixture() {}

void BestFitFixture::SetUp() { }
void BestFitFixture::TearDown() { }

void BestFitFixture::load_result(const std::string &path,std::vector<double> &out,std::vector<double> &rough_out) {
  TFile *file = TFile::Open(path.c_str());
  TTree *tree = static_cast<TTree*>(file->Get("fit_results"));
  if(!tree) {
    file->ls();
    throw GooStatsException("cannot find trees.");
  }
  out.clear();
  for(auto spc : species) {
    TTreeFormula *x = new TTreeFormula((spc+"["+std::to_string(subEntry)+"]").c_str(),
	(spc+"["+std::to_string(subEntry)+"]").c_str(),tree);
    tree->GetEntry(entry);
    x->GetNdata();
    out.push_back(x->EvalInstance());
    delete x;
  }
  for(auto rough_spc : rough_species) {
    TTreeFormula *x = new TTreeFormula((rough_spc+"["+std::to_string(subEntry)+"]").c_str(),
	(rough_spc+"["+std::to_string(subEntry)+"]").c_str(),tree);
    tree->GetEntry(entry);
    x->GetNdata();
    rough_out.push_back(x->EvalInstance());
    delete x;
  }
  file->Close();
}
