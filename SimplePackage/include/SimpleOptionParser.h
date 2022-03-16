/*****************************************************************************/
// Author: Xuefeng Ding <xuefengd@princeton.edu>
// Insitute: Department of Physics, Princeton University
// Date: 2/12/22
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2022 copyrighted.
/*****************************************************************************/

#ifndef GOOSTATS_SIMPLEOPTIONPARSER_H
#define GOOSTATS_SIMPLEOPTIONPARSER_H

#include "IOptionParser.h"

class SimpleOptionParser : public IOptionParser {
 public:
  bool parse(OptionManager *, const std::string &fileName) final;
  bool parse(OptionManager *, int argc, const char *argv[]) final;
};

#endif  //GOOSTATS_SIMPLEOPTIONPARSER_H
