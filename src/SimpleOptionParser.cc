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
  PrintAllOptions();
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
    if(!has(key)) 
      options.insert(std::make_pair(key,value));
    else {
      std::cout<<"Warning: duplicate term <"<<key<<"> found"
	<<" old: <"<<options.at(key)<<"> -> new: <"<<value<<">"<<std::endl;
      options.insert(std::make_pair(key,value));
    }
  }
  fin.close();
}

void SimpleOptionParser::PrintAllOptions() const {
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
