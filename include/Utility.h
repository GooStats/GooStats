/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef Utility_H
#define Utility_H
#include <string>
#include <vector>
namespace GooStats {
  namespace Utility {
    // naive splitter
    static std::vector<std::string> splitter(std::string source, std::string flag) {
      std::vector<std::string> result;
      while(source.find(flag)!=std::string::npos) {
	int position = source.find(flag);
	result.push_back(source.substr(0,position));
	source = source.substr(position+1,source.size()-position-1);
      }
      result.push_back(source);
      return result;
    };
  }
}
#endif
