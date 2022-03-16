/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "PdfCache.h"
#include "goofit/Variable.h"
DEVICE_VECTOR<fptype>* PdfCache_dev_vec[100];
MEM_CONSTANT fptype* PdfCache_dev_array[100];
PdfCache* PdfCache::cache = nullptr;
PdfCache* PdfCache::get() {
  if (!cache)
    cache = new PdfCache();
  return cache;
}
int PdfCache::registerFunc(PdfBase* pdf) {
  static int pdf_Id = 0;
  if (funMap.find(pdf) == funMap.end()) {
    assert(pdf_Id < 100);
    funMap.insert(std::make_pair(pdf, pdf_Id));
    PdfCache_dev_vec[pdf_Id] = new DEVICE_VECTOR<fptype>((*(pdf->obsCBegin()))->numbins);
    static fptype* dev_address[1];
    dev_address[0] = thrust::raw_pointer_cast(PdfCache_dev_vec[pdf_Id]->data());
    MEMCPY_TO_SYMBOL(
        PdfCache_dev_array, dev_address, sizeof(fptype*), pdf_Id * sizeof(fptype*), cudaMemcpyHostToDevice);
    printf("PdfCache::registerFunc register [%p](%s) as [%d]\n", pdf, pdf->getName().c_str(), funMap.at(pdf));
    pdf_Id++;
  }
  return funMap.at(pdf);
}
