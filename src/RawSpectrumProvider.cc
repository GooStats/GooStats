/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "RawSpectrumProvider.h"
#include "GooStatsException.h"
#include <iostream>
bool RawSpectrumProvider::registerSpecies(const std::string &name,int n_,const double *real_,double e0_,double de_) {
  if(n_map.find(name)!=n_map.end()) {
    std::cerr<<"Try to add ["<<name<<"] which already exists"<<std::endl;
    //throw GooStatsException("RawSpectrumProvider::registerSpecies Duplicate entries");
    return false;
  }
  std::cout<<"RawSpectrumProvider::registerSpecies Register ["<<name<<"]"<<std::endl;
  n_map.insert(make_pair(name,n_));
  real_map.insert(make_pair(name,real_));
  e0_map.insert(make_pair(name,e0_));
  de_map.insert(make_pair(name,de_));
  return true;
}
bool RawSpectrumProvider::registerComplexSpecies(const std::string &name,const std::map<std::string,double> &br_) {
  br_map.insert(make_pair(name,br_));
  return true;
}
int RawSpectrumProvider::n(const std::string &name) const { 
  if(n_map.find(name)!=n_map.end())
    return n_map.at(name); 
  else {
    std::cerr<<"trying to fetch ["<<name<<"] while RawSpectrumProvider does not have it"<<std::endl;
    throw GooStatsException("Raw array not ready");
  }
}
double const* RawSpectrumProvider::pdf(const std::string &name) const { 
  if(real_map.find(name)!=real_map.end())
    return real_map.at(name); 
  else {
    std::cerr<<"trying to fetch ["<<name<<"] while RawSpectrumProvider does not have it"<<std::endl;
    throw GooStatsException("Raw array not ready");
  }
}
double RawSpectrumProvider::e0(const std::string &name) const { 
  if(e0_map.find(name)!=e0_map.end())
    return e0_map.at(name); 
  else {
    std::cerr<<"trying to fetch ["<<name<<"] while RawSpectrumProvider does not have it"<<std::endl;
    throw GooStatsException("Raw array not ready");
  }
}
double RawSpectrumProvider::de(const std::string &name) const { 
  if(de_map.find(name)!=de_map.end())
    return de_map.at(name); 
  else {
    std::cerr<<"trying to fetch ["<<name<<"] while RawSpectrumProvider does not have it"<<std::endl;
    throw GooStatsException("Raw array not ready");
  }
}
double RawSpectrumProvider::peakE(const std::string &name) const { 
  if(peakE_map.find(name)!=peakE_map.end())
    return peakE_map.at(name); 
  else {
    std::cerr<<"trying to fetch ["<<name<<"] while RawSpectrumProvider does not have it"<<std::endl;
    throw GooStatsException("Raw array not ready");
  }
}
