/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef PdfCache_H
#define PdfCache_H
#include <map>

#include "goofit/PdfBase.h"
class PdfBase;
extern DEVICE_VECTOR<fptype> *PdfCache_dev_vec[100];
extern MEM_CONSTANT fptype *PdfCache_dev_array[100];
class PdfCache {
 public:
  static PdfCache *get();
  int registerFunc(PdfBase *pdf);

 private:
  PdfCache() {}
  static PdfCache *cache;
  std::map<PdfBase *, int> funMap;
};
#endif
