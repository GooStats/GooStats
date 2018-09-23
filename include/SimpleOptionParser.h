/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef SimpleOptionParser_H
#define SimpleOptionParser_H
#include "OptionManager.h"
#include <map>
#include <string>
class SimpleOptionParser : public OptionManager {
  public:
    bool parse(const std::string &fileName) final;
    bool parse(int argc,char **argv) final;
    std::string query(const std::string &key) const final;
    bool has(const std::string &key) const final;
    //! yes(key): if user forgot to put key, program will throw
    bool yes(const std::string &key) const final;
    //! hasAndYes(key): if user forgot to put key, program return false
    bool hasAndYes(const std::string &key) const final;
    void printAllOptions() const final;
  private:
    void do_parse(const std::string &fileName);
    void do_parse(int argc,char **argv);
    void insertKeyValue(const std::string &key,const std::string &value,bool allowOverwrite=false);
    std::map<std::string,std::string> options;
};
#endif
