/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "OptionManager.h"
#include <iostream>
#include "GooStatsException.h"

#define DEFINE_DatasetManager(T,var,ZERO) \
template<> T OptionManager::get(const std::string &key,bool thr) const { \
  if(var.find(key)==var.end()) { \
    if(thr) throw GooStatsException(key+" not found"); \
    else return ZERO; \
  } \
  return var.at(key); \
} \
template<> void OptionManager::set(const std::string &key,const T &value,bool allowOverwrite) { \
  if(var.find(key)!=var.end() && !allowOverwrite) {  \
    std::cout<<"Duplicate term <"<<key<<"> found" \
    <<" old: <"<<var.at(key)<<"> -> new: <"<<value<<">"<<std::endl; \
  } \
  var[key] = value; \
} 
bool OptionManager::has(const std::string &key) const { 
  return m_str.find(key)!=m_str.end() || m_double.find(key)!=m_double.end() || m_long.find(key)!=m_long.end();
} 
DEFINE_DatasetManager(std::string,m_str,"");
DEFINE_DatasetManager(double,m_double,0);
DEFINE_DatasetManager(long,m_long,0);
//    if(!allowOverwrite) 
//    throw GooStatsException("duplicate key inserted"); 

void OptionManager::printAllOptions() const {
  std::cout<<"*********Dump options parsed*********************************"<<std::endl;
  for(auto pair: m_str) 
    std::cout<<""<<(pair.first)<<" => <"<<(pair.second)<<">"<<std::endl;
  for(auto pair: m_double)  
    std::cout<<""<<(pair.first)<<" => float: ["<<pair.second<<"]"<<std::endl;
  for(auto pair: m_long)  
    std::cout<<""<<(pair.first)<<" => integer: ["<<pair.second<<"]"<<std::endl;
  std::cout<<"*************************************************************"<<std::endl;
}

#include <fstream>
#include <sstream>
bool OptionManager::parse(const std::string &filename) {
  if(filename.find("=")!=std::string::npos) { const char *c[] = {"",filename.c_str()}; parse(2,c); return true; }
  std::cout<<"Loading from <"<<filename<<">"<<std::endl;
  std::ifstream fin(filename.c_str());
  if(!fin.is_open())
    throw GooStatsException("ERROR: Unable to open config file");
  std::string key, value, equals, line;
  while( getline(fin,line) ) {
    if(line[0] == '#' || line.substr(0,2)=="//")
      continue;
    std::stringstream lstream(line);
    lstream >> key >> equals >> value;
    if(lstream.fail()) continue;
    if(equals != "=") continue;
    //std::cout<<"Parsing ["<<line<<"] -> ("<<key<<") ("<<equals<<") ("<<value<<")"<<std::endl;
    try {
      set(key,std::stod(value)); 
    } catch(...) {
      set(key,value);
    }
  }
  fin.close();
  return true;
}
bool OptionManager::parse(int argc,const char *argv[]) {
  for(int i = 1;i<argc;++i) {
    std::string item(argv[i]);
    auto eqPos = item.find("=");
    if(eqPos>0 && eqPos!=std::string::npos) {
      const std::string key = item.substr(0,eqPos);
      const std::string value = item.substr(eqPos+1);
      std::cout<<"ARG["<<i<<"] <"<<key<<"> = ["<<value<<"]"<<std::endl;
      try {
        set(key,std::stod(value)); 
      } catch(...) {
        set(key,value);
      }
    }
  }
  return true;
}
bool OptionManager::yes(const std::string &key) const {
  return (get(key)=="yes" || get(key)=="on" || get(key)=="true");
}
bool OptionManager::hasAndYes(const std::string &key) const {
  return has(key) && yes(key);
}
