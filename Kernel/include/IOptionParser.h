/*****************************************************************************/
// Author: Xuefeng Ding <xuefengd@princeton.edu>
// Institute: Department of Physics, Princeton University
// Date: 2/12/22
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2022 copyrighted.
/*****************************************************************************/

#ifndef GOOSTATS_IOPTIONPARSER_H
#define GOOSTATS_IOPTIONPARSER_H

#include <string>
class OptionManager;

class IOptionParser {
 public:
  virtual bool parse(OptionManager *, const std::string &fileName) = 0;
  virtual bool parse(OptionManager *, int argc, const char *argv[]) = 0;
};

#endif  //GOOSTATS_IOPTIONPARSER_H
