/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "SimplePlotManager.h"
#include "DatasetManager.h"
#include "TCanvas.h"
#include "goofit/PDFs/SumPdf.h"
void SimplePlotManager::draw(int ,const std::vector<DatasetManager*> &datasets) {
  auto datasetsgroups = groupByName(datasets);
  size_t i = 0;
  for(auto group : datasetsgroups) {
    auto cc = drawSingleGroup(group.first,group.second);
    if(!cc) continue;
    toBeSaved.insert(cc);
    if(createPdf()) {
      std::string tail = "";
      if(datasetsgroups.size()>1) {
	if(i==0) 
	  tail = "(";
	else if(i==datasetsgroups.size()-1)
	  tail = ")";
      }
      cc->Print((outName()+".pdf"+tail).c_str(),("Title:"+group.first).c_str());
      ++i;
    }
  }
}
std::map<std::string,std::vector<DatasetManager*>>
SimplePlotManager::groupByName(const std::vector<DatasetManager*>& datasets) {
  std::map<std::string,std::vector<DatasetManager*>> groups;
  for(auto dataset : datasets) {
    if(!dynamic_cast<SumPdf*>(dataset->getLikelihood())) continue;
    const std::string &name(dataset->name());
    const std::string &groupName(name.substr(0,name.find(".")));
    groups[groupName].push_back(dataset);
    std::cout<<"SimplePlotManager::groupByName <"<<name<<"> appedned to <"<<groupName<<">"<<std::endl;
  }
  return groups;
}
