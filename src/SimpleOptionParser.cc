/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "SimpleOptionParser.h"
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <time.h>
#include <sstream>
#include "GooStatsException.h"

bool SimpleOptionParser::parse(const std::string &fileName) {
  do_parse(fileName);
  return true;
}
bool SimpleOptionParser::parse(int argc,char **argv) {
  do_parse(argc,argv);
  return true;
}
std::string SimpleOptionParser::query(const std::string &key) const {
  if(has(key)) return options.at(key);
    throw GooStatsException(key+" not found");
}
bool SimpleOptionParser::has(const std::string &key) const {
  return options.find(key)!=options.end();
}
bool SimpleOptionParser::yes(const std::string &key) const {
  return (query(key)=="yes" || query(key)=="on" || query(key)=="true");
}
bool SimpleOptionParser::hasAndYes(const std::string &key) const {
  return has(key) && yes(key);
}


#include <sstream>
void SimpleOptionParser::do_parse(const std::string& filename) {
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
    insertKeyValue(key,value);
  }
  fin.close();
}
void SimpleOptionParser::do_parse(int argc,char **argv) {
  for(int i = 1;i<argc;++i) {
    std::string item(argv[i]);
    auto eqPos = item.find("=");
    if(eqPos>0 && eqPos!=std::string::npos) {
      const std::string key = item.substr(0,eqPos);
      const std::string value = item.substr(eqPos+1);
      std::cout<<"ARG["<<i<<"] <"<<key<<"> = ["<<value<<"]"<<std::endl;
      insertKeyValue(key,value,true /*force over-wrie*/);
    }
  }
}

void SimpleOptionParser::printAllOptions() const {
  std::cout<<"*********Dump options parsed*********************************"<<std::endl;
  for(auto pair: options)  {
    std::cout<<""<<(pair.first)<<" => ";
    if(::atof(pair.second.c_str())!=0) 
      std::cout<<"number: ["<<::atof(pair.second.c_str())<<"]"<<std::endl;
    else
      std::cout<<"<"<<(pair.second)<<">"<<std::endl;
  }
  std::cout<<"*************************************************************"<<std::endl;
}
void SimpleOptionParser::insertKeyValue(const std::string &key,const std::string &value,bool allowOverwrite) {
  if(!allowOverwrite && has(key)) 
    std::cout<<"Warning: duplicate term <"<<key<<"> found"
      <<" old: <"<<options.at(key)<<"> -> new: <"<<value<<">"<<std::endl;
  options[key] = value;
}
