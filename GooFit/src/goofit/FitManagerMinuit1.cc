PdfBase* pdfPointer; 
FitManager* currGlue = 0; 
int numPars = 0; 
vector<Variable*> FitManager::vars; 
bool FitManager::minim_conv = false;
bool FitManager::hesse_conv = false;

void specialTddpPrint (double fun); 

FitManager::FitManager (PdfBase* dat) 
  : minuit(0)
  , overrideCallLimit(-1)
{
  pdfPointer = dat;
  currGlue = this; 
} 

FitManager::~FitManager () {
  if (minuit) delete minuit; 
}

void FitManager::setupMinuit () {
  vars.clear(); 
  pdfPointer->getParameters(vars); 

  numPars = vars.size();
  if (minuit) delete minuit;
  minuit = new TMinuit(numPars); 
  int maxIndex = 0; 
  int counter = 0; 
  for (std::vector<Variable*>::iterator i = vars.begin(); i != vars.end(); ++i) {
    minuit->DefineParameter(counter, (*i)->name.c_str(), (*i)->value, (*i)->error, (*i)->lowerlimit, (*i)->upperlimit); 
    if ((*i)->fixed) minuit->FixParameter(counter);
    counter++; 
    if (maxIndex < (*i)->getIndex()) maxIndex = (*i)->getIndex();
  }

  numPars = maxIndex+1; 
  pdfPointer->copyParams();   
  minuit->SetFCN(FitFun); 
  if(!static_cast<GooPdf*>(pdfPointer)->IsChisquareFit())
      minuit->SetErrorDef(0.5);
}

void FitManager::fit () {
  setupMinuit();
  runMigrad();
}

void FitManager::runMigrad () { 
  assert(minuit);
  host_callnumber = 0;
//  int ierflg_simplex;
  int ierflg_minimize;
//  int ierflg_migrad;
  int ierflg_hesse;
//  minuit->mnexcm("SIMPLEX", 0,0,ierflg_simplex);
  if (0 < overrideCallLimit) {
    //std::cout << "Calling MIGRAD with call limit " << overrideCallLimit << std::endl; 
//    std::cout << "Calling MINIMIZE with call limit " << overrideCallLimit << std::endl; 
    double plist[1];
    plist[0] = overrideCallLimit;
    minuit->mnexcm("MINIMIZE", plist, 1, ierflg_minimize);
    minim_conv = ierflg_minimize == 0;
  }
  else minuit->Migrad(); 
//  minuit->mnexcm("SEEK", 0,0,ierflg_hesse);
//  minuit->mnexcm("SIMPLEX", 0,0,ierflg_simplex);
  minuit->mnexcm("HESSE", 0,0,ierflg_hesse);
  hesse_conv = ierflg_hesse == 0;
//  int ierflg;
//  minuit->mnexcm("MINOS", 0,0,ierflg);
  if(minim_conv) 
    ; //cout<<"MINIMIZE exit sucessfully"<<endl;
  else
    cout<<"MINIMIZE did not exit normally"<<endl;
  if(hesse_conv) 
    ; //cout<<"HESSE exit sucessfully"<<endl;
  else
    cout<<"HESSE did not exit normally"<<endl;
}

void FitManager::getMinuitValues () const {
  int counter = 0; 
  for (std::vector<Variable*>::iterator i = vars.begin(); i != vars.end(); ++i) {
    minuit->GetParameter(counter++, (*i)->value, (*i)->error);
  }
}

void FitFun(int & __attribute__((__unused__)) npar, double * __attribute__((__unused__)) gin, double &fun, double *fp, int __attribute__((__unused__)) iflag) {
  vector<double> pars;
  // Notice that npar is number of variable parameters, not total. 
  pars.resize(numPars); 
  int counter = 0; 
  for (auto i : FitManager::vars) {
    if (::isnan(fp[counter])) cout << "Variable " << i->name << " " << i->index << " is NaN\n"; 
    pars[i->getIndex()] = fp[counter++] + i->blind; 
  }
  
  pdfPointer->copyParams(pars); 
  fun = pdfPointer->calculateNLL(); 
  host_callnumber++; 
#ifdef NLL_CHECK
  throw -1;
#endif

#ifdef PRINTCALLS
  specialTddpPrint(fun); 
#endif 
}
int FitManager::getNumFreePars() {
  return currGlue->minuit->GetNumFreePars();
}

#ifdef PRINTCALLS
void specialTddpPrint (double fun) {
  // Stupid amplitude-fit debugging method. 
  cout << "Function call " << host_callnumber << ": " << fun << "\n";
  currGlue->getMinuitValues();
  int varCount = 1; 
  for (std::vector<Variable*>::iterator v = vars.begin(); v != vars.end(); ++v) {
    if (!(*v)) cout << "Null!" << endl; 
    if ((*v)->fixed) continue; 

    const fptype _mD0 = 1.86484; 
    const fptype _mD02 = _mD0 *_mD0;
    const fptype _mD02inv = 1./_mD02; 
    double stupidSpecialModifier = 1; // Mikhail interprets some of the weights differently. 
    if (((*v)->name == "f0_980_amp_real") || 
	((*v)->name == "f0_980_amp_imag") ||
	((*v)->name == "f0_1370_amp_real") || 
	((*v)->name == "f0_1370_amp_imag") ||
	((*v)->name == "f0_1500_amp_real") || 
	((*v)->name == "f0_1500_amp_imag") ||
	((*v)->name == "f0_1710_amp_real") || 
	((*v)->name == "f0_1710_amp_imag") ||
	((*v)->name == "f0_600_amp_real") || 
	((*v)->name == "f0_600_amp_imag")) stupidSpecialModifier = -_mD02; 
    else if (((*v)->name == "f2_1270_amp_real") ||
	     ((*v)->name == "f2_1270_amp_imag")) stupidSpecialModifier = -_mD02inv; 
    else if (((*v)->name == "nonr_amp_real") ||
	     ((*v)->name == "nonr_amp_imag")) stupidSpecialModifier = -1; 

    cout.width(20); 
    cout << (*v)->name;
    cout.setf(ios_base::right,ios_base::adjustfield);
    cout.width(3);
    cout << varCount++;
    cout.setf(ios_base::right,ios_base::adjustfield); cout.precision(8);
    cout << "  ";         cout.width(12);
    cout << (*v)->value / stupidSpecialModifier;
    cout.setf(ios_base::right,ios_base::adjustfield); cout.precision(8);
    cout << "  ";         cout.width(12);
    cout << (*v)->error;
    cout << endl; 
  }

  cout << endl; 
}
#endif
