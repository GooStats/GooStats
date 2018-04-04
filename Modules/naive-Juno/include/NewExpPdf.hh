#ifndef NEWEXP_PDF_HH
#define NEWEXP_PDF_HH

#include "goofit/PDFs/GooPdf.h" 

class NewExpPdf : public GooPdf {
public:
  NewExpPdf (std::string n, Variable* _x, Variable* alpha, Variable* offset = 0); 



private:

};

#endif
