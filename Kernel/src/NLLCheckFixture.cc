#include "NLLCheckFixture.h"
#include "GooStatsException.h"
#include "TFile.h"

#include "TSystem.h"
NLLCheckFixture::NLLCheckFixture() {
  gSystem->Load("libGooStatsNLLCheck.so");
}
NLLCheckFixture::~NLLCheckFixture() {}

void NLLCheckFixture::SetUp() { }
void NLLCheckFixture::TearDown() { }

#include "GooStatsNLLCheck.h"
void NLLCheckFixture::load_result(const std::string &path,const std::string &type,std::vector<double> &LLs,double &finalLL) {
  TFile *file = TFile::Open(path.c_str());
  GooStatsNLLCheck *obj = static_cast<GooStatsNLLCheck*>(file->Get(type.c_str()));
  if(!obj) {
    file->ls();
    throw GooStatsException("cannot find the object.");
  }
  LLs = obj->get_totLL();
  finalLL = obj->get_finalLL();
  file->Close();
}
