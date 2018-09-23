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
  for(size_t i = 0;i<species.size();++i) {
    TTreeFormula *x = new TTreeFormula(species.at(i).c_str(),species.at(i).c_str(),tree);
    tree->GetEntry(0);
    out.push_back(x->EvalInstance());
    delete x;
  }
  for(size_t i = 0;i<rough_species.size();++i) {
    TTreeFormula *x = new TTreeFormula(rough_species.at(i).c_str(),rough_species.at(i).c_str(),tree);
    tree->GetEntry(0);
    rough_out.push_back(x->EvalInstance());
    delete x;
  }
  file->Close();
}
