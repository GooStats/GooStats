/*****************************************************************************/
// Author: Xuefeng Ding <xuefengd@princeton.edu>
// Insitute: Department of Physics, Princeton University
// Date: 2/12/22
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2022 copyrighted.
/*****************************************************************************/

#include "SimpleOptionParser.h"

#include "GooStatsException.h"
#include "OptionManager.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include "Utility.h"
#include <regex>

bool SimpleOptionParser::parse(OptionManager *optionManager, const std::string &filename) {
  if (filename.find("=") != std::string::npos) {
    const char *c[] = {"", filename.c_str()};
    return parse(optionManager, 2, c);
  }
  std::cout << "Loading from <" << filename << ">" << std::endl;
  std::ifstream fin(filename.c_str());
  if (!fin.is_open())
    throw GooStatsException("ERROR: Unable to open config file");
  std::string key, value, equals, line;
  while (getline(fin, line)) {
    line = GooStats::Utility::strip(line);
    auto eq = line.find('=');
    if(eq!=std::string::npos) {
      key = GooStats::Utility::strip(line.substr(0,eq));
      value = GooStats::Utility::strip(line.substr(eq+1));
      optionManager->set(key, value,false);
    }
    //std::cout<<"Parsing ["<<line<<"] -> ("<<key<<") ("<<equals<<") ("<<value<<")"<<std::endl;
  }
  fin.close();
  return true;
}

bool SimpleOptionParser::parse(OptionManager *optionManager, int argc, const char *argv[]) {
  for (int i = 0; i < argc; ++i) {
    std::string item(argv[i]);
    auto eqPos = item.find("=");
    if (eqPos > 0 && eqPos != std::string::npos) {
      const std::string key = item.substr(0, eqPos);
      const std::string value = item.substr(eqPos + 1);
      std::cout << "ARG[" << i << "] <" << key << "> = [" << value << "]" << std::endl;
      optionManager->set(key, value,false);
    } else {
      throw GooStatsException("Illegal format of command line options: [" + item + "]");
    }
  }
  return true;
}