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

#include <fstream>
#include <sstream>
#include <iostream>
#include "GooStatsException.h"
#include "OptionManager.h"

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
        if (line[0] == '#' || line.substr(0, 2) == "//")
            continue;
        std::stringstream lstream(line);
        lstream >> key >> equals >> value;
        if (lstream.fail()) continue;
        if (equals != "=") continue;
        //std::cout<<"Parsing ["<<line<<"] -> ("<<key<<") ("<<equals<<") ("<<value<<")"<<std::endl;
        optionManager->set(key, value);
    }
    fin.close();
    return true;
}

bool SimpleOptionParser::parse(OptionManager *optionManager, int argc, const char *argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string item(argv[i]);
        auto eqPos = item.find("=");
        if (eqPos > 0 && eqPos != std::string::npos) {
            const std::string key = item.substr(0, eqPos);
            const std::string value = item.substr(eqPos + 1);
            std::cout << "ARG[" << i << "] <" << key << "> = [" << value << "]" << std::endl;
            optionManager->set(key, value);
        } else {
          this->parse(optionManager,item);
        }
    }
    return true;
}