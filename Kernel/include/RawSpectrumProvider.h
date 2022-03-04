/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef RawSpectrumProvider_H
#define RawSpectrumProvider_H
#include <map>
#include <string>
#include <vector>
// Protocol for raw spectrum provider
class RawSpectrumProvider {
  public:
    bool registerSpecies(const std::string &name,int n_, const double *real_,double e0_,double de_);
    bool linkSpecies(const std::string &target,const std::string &source);
    bool registerPeak(const std::string &name,double peakE_);
    bool registerComplexSpecies(const std::string &name,const std::map<std::string,double> &br_);
    size_t n(const std::string &name) const;
    const std::vector<double> &pdf(const std::string &name) const;
    double e0(const std::string &name) const; // in keV
    double de(const std::string &name) const; // in keV
    double peakE(const std::string &name) const; // in keV
    const std::map<std::string, std::map<std::string, double>> &get_br_map() const;
   private:
    std::map<std::string, int> n_map;
    std::map<std::string, std::vector<double>> real_map;
    std::map<std::string, double> e0_map;
    std::map<std::string, double> de_map;
    std::map<std::string, double> peakE_map;
    std::map<std::string, std::map<std::string, double> > br_map;
};
#endif
