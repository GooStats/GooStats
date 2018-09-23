/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef NeutrinoOscillationPdf_H
#define NeutrinoOscillationPdf_H

#include "goofit/PDFs/GooPdf.h" 

class NeutrinoOscillationPdf : public GooPdf {
public:
  NeutrinoOscillationPdf (std::string n, Variable *eNeu, std::vector<Variable*> sinThetas_2, std::vector<Variable*>deltaM2s, fptype distance);
    __host__ virtual fptype normalise () const;



private:

};

#endif
