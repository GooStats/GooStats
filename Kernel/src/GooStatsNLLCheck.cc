#include "GooStatsNLLCheck.h"
#include "TFile.h"
ClassImp(GooStatsNLLCheck)
GooStatsNLLCheck *GooStatsNLLCheck::me = nullptr;
GooStatsNLLCheck *GooStatsNLLCheck::get() {
  if(!me) me = new GooStatsNLLCheck();
  return me;
}
void GooStatsNLLCheck::init(const std::string &fname,const std::string &name) {
  file = TFile::Open(fname.c_str(),"RECREATE");
  this->SetName(name.c_str());
  this->SetTitle(name.c_str());
}
void GooStatsNLLCheck::new_LL(double _totLL) {
  totLL.push_back(_totLL-_s_totLL);
  _s_totLL = _totLL;
  results.push_back(result);
  result.clear();
}
void GooStatsNLLCheck::new_LL_single(double _singleLL) {
  totLL.push_back(_singleLL);
  _s_totLL += _singleLL;
  results.push_back(result);
  result.clear();
}
void GooStatsNLLCheck::record_LL(int bin,double E,double M,double T,double LL) {
  result[bin].E = E;
  result[bin].M = M;
  result[bin].T = T;
  result[bin].LL = LL;
}
void GooStatsNLLCheck::record_species(int bin,const std::string &name,double T) {
  result[bin].compositions[name] = T;
}
void GooStatsNLLCheck::save() {
  file->cd();
  this->Write();
  file->Close();
  delete file;
  file = nullptr;
}
void GooStatsNLLCheck::print() const {
#if __cplusplus <= 199711L
  for(size_t i = 0;i<results.size();++i) {
    const std::map<int,Info> &result = results.at(i);
    std::map<int,Info>::const_iterator binIt = result.begin();
    for(;binIt != result.end(); ++binIt) {
      printf("log(L) %.12le b %lf M %lf tot %.12le\n",binIt->second.LL,binIt->second.E,binIt->second.M,binIt->second.T);
      std::map<std::string,double>::const_iterator spcIt = binIt->second.compositions.begin();
      for(;spcIt != binIt->second.compositions.end(); ++spcIt) {
        printf(" %s %.12le",spcIt->first.c_str(),spcIt->second);
      }
      printf("\n");
    }
  }
  for(size_t i = 0;i<totLL.size();++i) {
    printf("log(L) %.12le\n",totLL.at(i));
  }
  printf("final log(L) %.12le\n",finalLL);
#else
  for(auto ele : get_results()) {
    for(auto bin : ele) {
      printf("log(L) %.12le b %lf M %lf tot %.12le\n",bin.second.LL,bin.second.E,bin.second.M,bin.second.T);
      for(auto spc : bin.second.compositions) {
        printf(" %s %.12le",spc.first.c_str(),spc.second);
      }
      printf("\n");
    }
  }
  for(auto LL: get_totLL()) {
    printf("log(L) %.12le\n",LL);
  }
  printf("final log(L) %.12le\n",finalLL);
#endif
}
