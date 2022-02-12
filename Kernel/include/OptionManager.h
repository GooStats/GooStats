/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef OptionManager_H
#define OptionManager_H
#include <string>
#include <map>
// protocol for option configset class
class OptionManager {
  public:
    // can be fileName or key=value sentence
    template<typename T = std::string> T get(const std::string &,bool=true) const;
    void set(const std::string &,const std::string &,bool = false);
    bool has(const std::string &) const;
    //! yes(key): if user forgot to put key, program will throw
    bool yes(const std::string &key) const;
    //! hasAndYes(key): if user forgot to put key, program return false
    bool hasAndYes(const std::string &key) const;
    void printAllOptions() const;
  private:
    std::map<std::string,std::string> m_data;
};



#endif
