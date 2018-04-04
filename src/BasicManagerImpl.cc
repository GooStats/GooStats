/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "BasicManagerImpl.h"
#include <iostream>
#include "goofit/Variable.h"
#include "GooStatsException.h"
Variable *BasicManagerImpl::createVar(const std::string &key,double val,double err,double min,double max) { 
  if(!hasVar(key)) {
    std::shared_ptr<Variable> var_(new Variable(m_name+"."+key,val,err,min,max));
    std::cout<<"Inserting ["<<key<<"]("<<var_.get()<<") to ["<<m_name<<"]"<<std::endl; 
    m_var.insert(std::make_pair(key,var_));
  }
  return m_var.at(key).get();
}
Variable *BasicManagerImpl::linkVar(const std::string &key,const std::string &source) { 
  if(!hasVar(key)) std::cout<<"Linking ["<<key<<"] with ["<<source<<"] in ["<<m_name<<"]"<<std::endl; 
  std::shared_ptr<Variable> var_(m_var.at(source));
  m_var.insert(std::make_pair(key,var_));
  return m_var.at(key).get();
}
bool BasicManagerImpl::hasVar(const std::string &key) const {
  return m_var.find(key)!=m_var.end();
}
Variable *BasicManagerImpl::var(const std::string &key) const {
  try {
    return m_var.at(key).get();
  } catch( std::out_of_range &ex ) {
    std::cerr<<"Trying to access ["<<key<<"] while ["<<m_name<<"] does not have it"<<std::endl;
    throw GooStatsException("Variable not available");
  }
}
