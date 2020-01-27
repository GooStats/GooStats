#include "goofit/GlobalCudaDefines.h"
#include "goofit/PDFs/GooPdf.h"
#include "thrust/sequence.h" 
#include "thrust/iterator/constant_iterator.h" 
#include <fstream> 

// These variables are either function-pointer related (thus specific to this implementation)
// or constrained to be in the CUDAglob translation unit by nvcc limitations; otherwise they 
// would be in PdfBase. 

// Device-side, translation-unit constrained. 
MEM_CONSTANT fptype cuda_array[maxParams];           // Holds device-side fit parameters. 
MEM_DEVICE unsigned int paramIndices[maxIndicies];  // Holds functor-specific indices into cuda_array. Also overloaded to hold integer constants (ie parameters that cannot vary.) 
MEM_DEVICE fptype functorConstants[maxConsts];    // Holds non-integer constants. Notice that first entry is number of events. 
MEM_CONSTANT fptype normalisationFactors[maxParams]; 

// For debugging 
MEM_CONSTANT int callnumber; 
MEM_CONSTANT int gpuDebug; 
MEM_CONSTANT unsigned int debugParamIndex;
MEM_DEVICE int internalDebug1 = -1; 
MEM_DEVICE int internalDebug2 = -1; 
MEM_DEVICE int internalDebug3 = -1; 
int cpuDebug = 0; 
#ifdef PROFILING
MEM_DEVICE fptype timeHistogram[10000]; 
fptype host_timeHist[10000];
#endif 

// Function-pointer related. 
MEM_DEVICE void* device_function_table[200]; // Not clear why this cannot be MEM_CONSTANT, but it causes crashes to declare it so. 
void* host_function_table[200];
unsigned int num_device_functions = 0; 
map<void*, int> functionAddressToDeviceIndexMap; 

// For use in debugging memory issues
void printMemoryStatus (std::string file, int line) {
  size_t memfree = 0;
  size_t memtotal = 0; 
  SYNCH(); 
// Thrust 1.7 will make the use of THRUST_DEVICE_BACKEND an error
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  cudaMemGetInfo(&memfree, &memtotal); 
#endif
  SYNCH(); 
  std::cout << "Memory status " << file << " " << line << " Free " << memfree << " Total " << memtotal << " Used " << (memtotal - memfree) << std::endl;
}


#include <execinfo.h>
void* stackarray[10];
void abortWithCudaPrintFlush (std::string file, int line, std::string reason, const PdfBase* pdf ) {
#ifdef CUDAPRINT
  cudaPrintfDisplay(stdout, true);
  cudaPrintfEnd();
#endif
  std::cout << "Abort called from " << file << " line " << line << " due to " << reason << std::endl; 
  if (pdf) {
    PdfBase::parCont pars;
    pdf->getParameters(pars);
    std::cout << "Parameters of " << pdf->getName() << " : \n";
    for (PdfBase::parIter v = pars.begin(); v != pars.end(); ++v) {
      if (0 > (*v)->index) continue; 
      std::cout << "  " << (*v)->name << " (" << (*v)->index << ") :\t" << host_params[(*v)->index] << std::endl;
    }
  }

  std::cout << "Parameters (" << totalParams << ") :\n"; 
  for (int i = 0; i < totalParams; ++i) {
    std::cout << host_params[i] << " ";
  }
  std::cout << std::endl; 


  // get void* pointers for all entries on the stack
  size_t size = backtrace(stackarray, 10);
  // print out all the frames to stderr
  backtrace_symbols_fd(stackarray, size, 2);

  exit(1); 
}

EXEC_TARGET fptype calculateEval (fptype rawPdf, fptype* evtVal, unsigned int par) {
  // Just return the raw PDF value, for use in (eg) normalisation. 
  return rawPdf; 
}

EXEC_TARGET fptype calculateNLL (fptype rawPdf, fptype* evtVal, unsigned int par) {
//  rawPdf *= normalisationFactors[par];
  return -EVALLOG(rawPdf);
}

EXEC_TARGET fptype calculateProb (fptype rawPdf, fptype* evtVal, unsigned int par) {
  // Return probability, ie normalised PDF value.
//  rawPdf *= normalisationFactors[par];
  return rawPdf; 
}

EXEC_TARGET fptype calculateBinAvg (fptype rawPdf, fptype* evtVal, unsigned int par) {
//  rawPdf *= normalisationFactors[par];
  rawPdf *= evtVal[1]; // Bin volume 
  // Log-likelihood of numEvents with expectation of exp is (-exp + numEvents*ln(exp) - ln(numEvents!)). 
  // The last is constant, so we drop it; and then multiply by minus one to get the negative log-likelihood. 
#ifdef CUDADEBUG
  printf("BID %d(%d) TID %d(%d) npe %lf M %lf T %lf -log(L) %.1le\n",
      BLOCKIDX,gridDim.x,THREADIDX,BLOCKDIM,
      evtVal[-1],evtVal[0],rawPdf,
      (rawPdf- evtVal[0]*EVALLOG(rawPdf)+::lgamma(evtVal[0]+1)));
#endif
  //if (rawPdf > 0 && (evtVal[-1]>140 || evtVal[-1]<100)) {
  if (rawPdf > 0 ) {
    // functorConstants[0] is the number of events in the dataset
    const fptype expEvents = rawPdf;
    return (expEvents - evtVal[0]*EVALLOG(expEvents)+::lgamma(evtVal[0]+1)); 
  } 
  return 0; 
}

EXEC_TARGET fptype calculateScaledBinAvg (fptype rawPdf, fptype* evtVal, unsigned int par) {
  if (rawPdf > 0) {
    const fptype expEvents = rawPdf*evtVal[1]; // Bin volume
    const fptype measureEvents  = evtVal[0];
    fptype result;
    const fptype npe = evtVal[-1];
    const bool apply_scale = npe<RO_CACHE(dev_Nll_threshold[0]);
    if(apply_scale && (expEvents>10) ) {
      const fptype diff = measureEvents-expEvents;
      const fptype err_square = measureEvents*RO_CACHE(dev_Nll_scaleFactor[0]);
      const fptype pi = 3.1415926535897932384626433832795028841971693993751;
      result = 0.5*diff*diff/err_square + 0.5*log(2*pi*err_square);
    } else {
      result = (expEvents - measureEvents*EVALLOG(expEvents)+::lgamma(measureEvents+1)); 
    }
    return result;
  } 
  return 0; 
}

EXEC_TARGET fptype calculateBinWithError (fptype rawPdf, fptype* evtVal, unsigned int par) {
  // In this case interpret the rawPdf as just a number, not a number of events. 
  // Do not divide by integral over phase space, do not multiply by bin volume, 
  // and do not collect 200 dollars. evtVal should have the structure (bin entry, bin error). 
  //printf("[%i, %i] ((%f - %f) / %f)^2 = %f\n", BLOCKIDX, THREADIDX, rawPdf, evtVal[0], evtVal[1], POW((rawPdf - evtVal[0]) / evtVal[1], 2)); 
  rawPdf -= evtVal[0]; // Subtract observed value.
  rawPdf /= evtVal[1]; // Divide by error.
  rawPdf *= rawPdf; 
  return rawPdf; 
}

EXEC_TARGET fptype calculateChisq (fptype rawPdf, fptype* evtVal, unsigned int par) {
//  rawPdf *= normalisationFactors[par];
  rawPdf *= evtVal[1]; // Bin volume 
  if (evtVal[0] > 0) {
    double diff = rawPdf-evtVal[0];
    return diff*diff/evtVal[0];
  } 
  return 0; 
}

MEM_DEVICE device_metric_ptr ptr_to_Eval         = calculateEval; 
MEM_DEVICE device_metric_ptr ptr_to_NLL          = calculateNLL;  
MEM_DEVICE device_metric_ptr ptr_to_Prob         = calculateProb; 
MEM_DEVICE device_metric_ptr ptr_to_BinAvg       = calculateBinAvg;  
MEM_DEVICE device_metric_ptr ptr_to_ScaledBinAvg       = calculateScaledBinAvg;  
MEM_DEVICE device_metric_ptr ptr_to_BinWithError = calculateBinWithError;
MEM_DEVICE device_metric_ptr ptr_to_Chisq        = calculateChisq; 

void* host_fcn_ptr = 0;

void* getMetricPointer (std::string name) {
  #define CHOOSE_PTR(ptrname) if (name == #ptrname) GET_FUNCTION_ADDR(ptrname);
  host_fcn_ptr = 0; 
  CHOOSE_PTR(ptr_to_Eval); 
  CHOOSE_PTR(ptr_to_NLL); 
  CHOOSE_PTR(ptr_to_Prob); 
  CHOOSE_PTR(ptr_to_BinAvg); 
  CHOOSE_PTR(ptr_to_ScaledBinAvg); 
  CHOOSE_PTR(ptr_to_BinWithError); 
  CHOOSE_PTR(ptr_to_Chisq); 

  assert(host_fcn_ptr); 

  return host_fcn_ptr;
#undef CHOOSE_PTR
}


GooPdf::GooPdf (Variable* x, std::string n) 
  : PdfBase(x, n)
  , logger(0LL)
  , binnedlogger(0LL)
{
  //std::cout << "Created " << n << std::endl; 
}

__host__ int GooPdf::findFunctionIdx (void* dev_functionPtr) {
  // Code specific to function-pointer implementation 
  map<void*, int>::iterator localPos = functionAddressToDeviceIndexMap.find(dev_functionPtr); 
  if (localPos != functionAddressToDeviceIndexMap.end()) {
    return (*localPos).second; 
  }

  int fIdx = num_device_functions;   
  host_function_table[num_device_functions] = dev_functionPtr;
  functionAddressToDeviceIndexMap[dev_functionPtr] = num_device_functions; 
  num_device_functions++; 
  MEMCPY_TO_SYMBOL(device_function_table, host_function_table, num_device_functions*sizeof(void*), 0, cudaMemcpyHostToDevice); 

#ifdef PROFILING
  host_timeHist[fIdx] = 0; 
  MEMCPY_TO_SYMBOL(timeHistogram, host_timeHist, 10000*sizeof(fptype), 0,cudaMemcpyHostToDevice);
#endif 

  return fIdx; 
}

__host__ void GooPdf::initialise (std::vector<unsigned int> pindices, void* dev_functionPtr) {
  if (!fitControl) setFitControl(new UnbinnedNllFit()); 

  // MetricTaker must be created after PdfBase initialisation is done.
  PdfBase::initialiseIndices(pindices); 

  functionIdx = findFunctionIdx(dev_functionPtr); 
  setMetrics(); 
}

__host__ void GooPdf::setDebugMask (int mask, bool setSpecific) const {
  cpuDebug = mask; 
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
  gpuDebug = cpuDebug;
  if (setSpecific) debugParamIndex = parameters; 
#else
  MEMCPY_TO_SYMBOL(gpuDebug, &cpuDebug, sizeof(int), 0, cudaMemcpyHostToDevice);
  if (setSpecific) MEMCPY_TO_SYMBOL(debugParamIndex, &parameters, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
#endif
} 

__host__ void GooPdf::setMetrics () {
  delete logger;
  logger = new MetricTaker(this, getMetricPointer(fitControl->getMetric()));  
  delete binnedlogger;
  binnedlogger = new BinnedMetricTaker(this, getMetricPointer(fitControl->getMetric()));  
//  cout<<getName()<<" set metric to "<<fitControl->getMetric()<<endl;
}

__host__ double GooPdf::sumOfNll (int numVars) const {
  static thrust::plus<double> cudaPlus;
  thrust::constant_iterator<int> eventSize(numVars); 
  thrust::constant_iterator<fptype*> arrayAddress(dev_event_array[pdfId]); 
  double dummy = 0;
  thrust::counting_iterator<int> eventIndex(0); 
  return thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(eventIndex, arrayAddress, eventSize)),
				  thrust::make_zip_iterator(thrust::make_tuple(eventIndex + numEntries, arrayAddress, eventSize)),
				  *logger, dummy, cudaPlus);   
}

__host__ double GooPdf::calculateNLL () const {
  // this is the exact start of the GooFit minimization
  normalise();

  if (host_normalisation[parameters] <= 0) 
    abortWithCudaPrintFlush(__FILE__, __LINE__, getName() + " non-positive normalisation", this);
//
//  MEMCPY_TO_SYMBOL(normalisationFactors, host_normalisation, totalParams*sizeof(fptype), 0, cudaMemcpyHostToDevice); 
//  SYNCH(); // Ensure normalisation integrals are finished

  int numVars = observables.size(); 
  if (fitControl->binnedFit()) {
    numVars += 2;
    numVars *= -1; 
  }

  fptype ret = sumOfNll(numVars); 
//  if (0 == ret) abortWithCudaPrintFlush(__FILE__, __LINE__, getName() + " zero NLL", this); 
  return ret; 
}

__host__ void GooPdf::evaluateAtPoints (Variable* var, std::vector<fptype>& res) {
  // NB: This does not project correctly in multidimensional datasets, because all observables
  // other than 'var' will have, for every event, whatever value they happened to get set to last
  // time they were set. This is likely to be the value from the last event in whatever dataset
  // you were fitting to, but at any rate you don't get the probability-weighted integral over
  // the other observables. 

  copyParams(); 
  normalise(); 
//  MEMCPY_TO_SYMBOL(normalisationFactors, host_normalisation, totalParams*sizeof(fptype), 0, cudaMemcpyHostToDevice); 
  UnbinnedDataSet tempdata(observables);

  double step = (var->upperlimit - var->lowerlimit) / var->numbins; 
  for (int i = 0; i < var->numbins; ++i) {
    var->value = var->lowerlimit + (i+0.5)*step;
    tempdata.addEvent(); 
  }
  setData(&tempdata);  
 
  thrust::counting_iterator<int> eventIndex(0); 
  thrust::constant_iterator<int> eventSize(observables.size());
  thrust::constant_iterator<fptype*> arrayAddress(dev_event_array[pdfId]); 
  DEVICE_VECTOR<fptype> results(var->numbins); 

  MetricTaker evalor(this, getMetricPointer("ptr_to_Eval")); 
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(eventIndex, arrayAddress, eventSize)),
      thrust::make_zip_iterator(thrust::make_tuple(eventIndex + numEntries, arrayAddress, eventSize)),
      results.begin(),
      evalor); 

  thrust::host_vector<fptype> h_results = results; 
  res.clear();
  res.resize(var->numbins);
  for (int i = 0; i < var->numbins; ++i) {
    res[i] = h_results[i] * host_normalisation[parameters];
  }
}

__host__ void GooPdf::evaluateAtPoints (std::vector<fptype>& __attribute__((__unused__)) points) const {
  /*
  std::set<Variable*> vars;
  getParameters(vars);
  unsigned int maxIndex = 0;
  for (std::set<Variable*>::iterator i = vars.begin(); i != vars.end(); ++i) {
    if ((*i)->getIndex() < maxIndex) continue;
    maxIndex = (*i)->getIndex();
  }
  std::vector<double> params;
  params.resize(maxIndex+1);
  for (std::set<Variable*>::iterator i = vars.begin(); i != vars.end(); ++i) {
    if (0 > (*i)->getIndex()) continue;
    params[(*i)->getIndex()] = (*i)->value;
  } 
  copyParams(params); 

  DEVICE_VECTOR<fptype> d_vec = points; 
  normalise(); 
  //MEMCPY_TO_SYMBOL(normalisationFactors, host_normalisation, totalParams*sizeof(fptype), 0, cudaMemcpyHostToDevice); 
  thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), *evalor);
  thrust::host_vector<fptype> h_vec = d_vec;
  for (unsigned int i = 0; i < points.size(); ++i) points[i] = h_vec[i]; 
  */
}

__host__ void GooPdf::scan (Variable* var, std::vector<fptype>& values) {
  fptype step = var->upperlimit;
  step -= var->lowerlimit;
  step /= var->numbins;
  values.clear(); 
  for (fptype v = var->lowerlimit + 0.5*step; v < var->upperlimit; v += step) {
    var->value = v;
    copyParams();
    parCont pars;
    getParameters(pars);
    for(unsigned int j = 0;j<pars.size();++j)
      printf("p%d %p %lf ",j,pars.at(j),pars.at(j)->value);
    printf("\n");
    fptype curr = calculateNLL(); 
    values.push_back(curr);
  }
}

__host__ void GooPdf::setParameterConstantness (bool constant) {
  PdfBase::parCont pars; 
  getParameters(pars); 
  for (PdfBase::parIter p = pars.begin(); p != pars.end(); ++p) {
    (*p)->fixed = constant; 
  }
}

__host__ fptype GooPdf::getValue () {
  // Returns the value of the PDF at a single point. 
  // Execute redundantly in all threads for OpenMP multiGPU case
  copyParams(); 
  normalise(); 
  //MEMCPY_TO_SYMBOL(normalisationFactors, host_normalisation, totalParams*sizeof(fptype), 0, cudaMemcpyHostToDevice); 

  UnbinnedDataSet point(observables); 
  point.addEvent(); 
  setData(&point); 

  thrust::counting_iterator<int> eventIndex(0); 
  thrust::constant_iterator<int> eventSize(observables.size()); 
  thrust::constant_iterator<fptype*> arrayAddress(dev_event_array[pdfId]); 
  DEVICE_VECTOR<fptype> results(1); 
  
  MetricTaker evalor(this, getMetricPointer("ptr_to_Eval"));
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(eventIndex, arrayAddress, eventSize)),
		    thrust::make_zip_iterator(thrust::make_tuple(eventIndex + 1, arrayAddress, eventSize)),
		    results.begin(),
		    evalor); 
  return results[0];
}

__host__ fptype GooPdf::normalise () const {
  //if (cpuDebug & 1) std::cout << "Normalising " << getName() << " " << hasAnalyticIntegral() << " " << normRanges << std::endl;

  if (!fitControl->metricIsPdf()) {
    host_normalisation[parameters] = 1.0; 
    return 1.0;
  }

  fptype ret = 1;
  if (hasAnalyticIntegral()) {
    for (obsConstIter v = obsCBegin(); v != obsCEnd(); ++v) { // Loop goes only over observables of this PDF. 
      //if (1) std::cout << "Analytically integrating " << getName() << " over " << (*v)->name << std::endl; 
      // what is this doing?? how does the integrate know what variable you are integrating?? if this is only 1-D, why would you write the loop??
      ret *= integrate((*v)->lowerlimit, (*v)->upperlimit);
    }
    host_normalisation[parameters] = 1.0/ret;
    //if (1) std::cout << "Analytic integral of " << getName() << " is " << ret 
    //  <<" par: "<<parameters<<" "<<host_normalisation[0]<<" "<<host_normalisation[1]
    //  << std::endl; 

    return ret; 
  } 

  int totalBins = 1; 
  for (obsConstIter v = obsCBegin(); v != obsCEnd(); ++v) {
    ret *= ((*v)->upperlimit - (*v)->lowerlimit);
    totalBins *= (integrationBins > 0 ? integrationBins : (*v)->numbins); 
    //if (cpuDebug & 1) std::cout << "Total bins " << totalBins << " due to " << (*v)->name << " " << integrationBins << " " << (*v)->numbins << std::endl; 
  }
  ret /= totalBins; 

  fptype dummy = 0; 
  static thrust::plus<fptype> cudaPlus;
  thrust::constant_iterator<fptype*> arrayAddress(normRanges); 
  thrust::constant_iterator<int> eventSize(observables.size());
  thrust::counting_iterator<int> binIndex(0); 
  fptype sum = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(binIndex, eventSize, arrayAddress)),
					thrust::make_zip_iterator(thrust::make_tuple(binIndex + totalBins, eventSize, arrayAddress)),
					*binnedlogger, dummy, cudaPlus); 
 
  if (std::isnan(sum)) {
    abortWithCudaPrintFlush(__FILE__, __LINE__, getName() + " NaN in normalisation", this); 
  }
  else if (0 >= sum) { 
    abortWithCudaPrintFlush(__FILE__, __LINE__, "Non-positive normalisation", this); 
  }

  ret *= sum;


  if (0 == ret) abortWithCudaPrintFlush(__FILE__, __LINE__, "Zero integral"); 
  host_normalisation[parameters] = 1.0/ret;
  return (fptype) ret; 
}

#ifdef PROFILING
MEM_CONSTANT fptype conversion = (1.0 / CLOCKS_PER_SEC); 
EXEC_TARGET fptype callFunction (fptype* eventAddress, unsigned int functionIdx, unsigned int paramIdx) {
  clock_t start = clock();
  fptype ret = (*(reinterpret_cast<device_function_ptr>(device_function_table[functionIdx])))(eventAddress, cuda_array, paramIndices + paramIdx);
  clock_t stop = clock(); 
  if ((0 == THREADIDX + BLOCKIDX) && (stop > start)) {
    // Avoid issue when stop overflows and start doesn't. 
    timeHistogram[functionIdx*100 + paramIdx] += ((stop - start) * conversion); 
    //printf("Clock: %li %li %li | %u %f\n", (long) start, (long) stop, (long) (stop - start), functionIdx, timeHistogram[functionIdx]); 
  }
  return ret; 
}
#else 
EXEC_TARGET fptype callFunction (fptype* eventAddress, unsigned int functionIdx, unsigned int paramIdx) {
  return (*(reinterpret_cast<device_function_ptr>(device_function_table[functionIdx])))(eventAddress, cuda_array, paramIndices + paramIdx);
}
#endif 

// Notice that operators are distinguished by the order of the operands,
// and not otherwise! It's up to the user to make his tuples correctly. 

// Main operator: Calls the PDF to get a predicted value, then the metric 
// to get the goodness-of-prediction number which is returned to MINUIT. 
EXEC_TARGET fptype MetricTaker::operator () (thrust::tuple<int, fptype*, int> t) const {
  // Calculate event offset for this thread. 
  int eventIndex = thrust::get<0>(t);
  int eventSize  = thrust::get<2>(t);
  fptype* eventAddress = thrust::get<1>(t) + (eventIndex * abs(eventSize)); 
  fptype ret = callFunction(eventAddress, functionIdx, parameters);
#ifdef CUDADEBUG
  printf("BID %d(%d) TID %d(%d) binSize %d npe %lf Nevents %lf npe_binSize %lf\n",
      BLOCKIDX,gridDim.x,THREADIDX,BLOCKDIM,
      eventSize,eventAddress[0],eventAddress[1],eventAddress[2]);
#endif

  // Notice assumption here! For unbinned fits the 'eventAddress' pointer won't be used
  // in the metric, so it doesn't matter what it is. For binned fits it is assumed that
  // the structure of the event is (obs1 obs2... binentry binvolume), so that the array
  // passed to the metric consists of (binentry binvolume). 
  ret = (*(reinterpret_cast<device_metric_ptr>(device_function_table[metricIndex])))(ret, eventAddress + (abs(eventSize)-2), parameters);
  return ret; 
}
 
// Operator for binned evaluation, no metric. 
// Used in normalisation. 
#define MAX_NUM_OBSERVABLES 2
EXEC_TARGET fptype BinnedMetricTaker::operator () (thrust::tuple<int, int, fptype*> t) const {
  // Bin index, event size, base address [lower, upper, numbins] 
 
  int binNumber = thrust::get<0>(t);
  const int evtSize = thrust::get<1>(t);
  
  MEM_SHARED fptype binCenters[1024*MAX_NUM_OBSERVABLES]; // cannot use the stack memory..  don't know why

  // To convert global bin number to (x,y,z...) coordinates: For each dimension, take the mod 
  // with the number of bins in that dimension. Then divide by the number of bins, in effect
  // collapsing so the grid has one fewer dimension. Rinse and repeat. 
  unsigned int const* const indices = paramIndices + parameters;
  for (int i = 0; i < evtSize; ++i) {
    const fptype lowerBound = thrust::get<2>(t)[3*i+0];
    const fptype upperBound = thrust::get<2>(t)[3*i+1];
    const int numBins    = (int) FLOOR(thrust::get<2>(t)[3*i+2] + 0.5); 
    const int localBin = binNumber % numBins;

    fptype x = upperBound - lowerBound; 
    x /= numBins;
    x *= (localBin + 0.5); 
    x += lowerBound;
    binCenters[RO_CACHE(indices[RO_CACHE(indices[0]) + 2 + i])+THREADIDX*MAX_NUM_OBSERVABLES] = x; 
    binNumber /= numBins;
  }
  return callFunction(binCenters+THREADIDX*MAX_NUM_OBSERVABLES, functionIdx, parameters); 
}

__host__ void GooPdf::getCompProbsAtDataPoints (std::vector<std::vector<fptype> >& values) {
  copyParams(); 
  double overall = normalise();
//  MEMCPY_TO_SYMBOL(normalisationFactors, host_normalisation, totalParams*sizeof(fptype), 0, cudaMemcpyHostToDevice); 

  int numVars = observables.size(); 
  if (fitControl->binnedFit()) {
    numVars += 2;
    numVars *= -1; 
  }
  DEVICE_VECTOR<fptype> results(numEntries); 
  thrust::constant_iterator<int> eventSize(numVars); 
  thrust::constant_iterator<fptype*> arrayAddress(dev_event_array[pdfId]); 
  thrust::counting_iterator<int> eventIndex(0); 
  MetricTaker evalor(this, getMetricPointer("ptr_to_Prob")); 
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(eventIndex, arrayAddress, eventSize)),
		    thrust::make_zip_iterator(thrust::make_tuple(eventIndex + numEntries, arrayAddress, eventSize)),
		    results.begin(), 
		    evalor); 
  values.clear(); 
  values.resize(components.size() + 1);
  thrust::host_vector<fptype> host_results = results;
  for (unsigned int i = 0; i < host_results.size(); ++i) {
    values[0].push_back(host_results[i]);
  }
  
  for (unsigned int i = 0; i < components.size(); ++i) {
    MetricTaker compevalor(components[i], getMetricPointer("ptr_to_Prob")); 
    thrust::counting_iterator<int> ceventIndex(0); 
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(ceventIndex, arrayAddress, eventSize)),
		      thrust::make_zip_iterator(thrust::make_tuple(ceventIndex + numEntries, arrayAddress, eventSize)),
		      results.begin(), 
		      compevalor); 
    host_results = results;
    for (unsigned int j = 0; j < host_results.size(); ++j) {
      values[1 + i].push_back(host_results[j]); 
    }    
  }
}

// still need to add OpenMP/multi-GPU code here
__host__ void GooPdf::transformGrid (fptype* host_output) { 
  generateNormRange(); 
  //normalise(); 
  unsigned int totalBins = 1; 
  for (obsConstIter v = obsCBegin(); v != obsCEnd(); ++v) {
    totalBins *= (*v)->numbins; 
  }

  thrust::constant_iterator<fptype*> arrayAddress(normRanges); 
  thrust::constant_iterator<int> eventSize(observables.size());
  thrust::counting_iterator<int> binIndex(0); 
  DEVICE_VECTOR<fptype> d_vec;
  d_vec.resize(totalBins); 

  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(binIndex, eventSize, arrayAddress)),
		    thrust::make_zip_iterator(thrust::make_tuple(binIndex + totalBins, eventSize, arrayAddress)),
		    d_vec.begin(), 
		    *binnedlogger); 

  thrust::host_vector<fptype> h_vec = d_vec;
  for (unsigned int i = 0; i < totalBins; ++i) host_output[i] = h_vec[i]; 
}


__host__ void GooPdf::debug() const {
  unsigned int *indices;
  unsigned int *devcpy_indices = new unsigned int[maxParams];
  MEMCPY_FROM_SYMBOL(devcpy_indices,paramIndices,maxParams*sizeof(unsigned int),0,cudaMemcpyDeviceToHost);
  indices = devcpy_indices+parameters;
  cout<<getName()<<"  initial condition: id";
  int totpar = indices[0];
  int totobs = indices[indices[0]+1];
  cout<<"-(totpar)"<<totpar;
  for(int i = 0;i<totpar;++i)
    cout<<"-(par"<<i<<")"<<indices[i+1];
  cout<<"-(totobs)"<<totobs;
  for(int i = 0;i<totobs;++i)
    cout<<"-(obs"<<i<<")"<<indices[i+totpar+2];
  cout<<endl;
  indices = (unsigned int *)0;
  delete [] devcpy_indices;

  indices = host_indices + parameters;
  totpar = indices[0];
  totobs = indices[indices[0]+1];
  cout<<getName()<<" from host ini cond: id";
  cout<<"-(totpar)"<<totpar;
  for(int i = 0;i<totpar;++i)
    cout<<"-(par"<<i<<")"<<indices[i+1];
  cout<<"-(totobs)"<<totobs;
  for(int i = 0;i<totobs;++i)
    cout<<"-(obs"<<i<<")"<<indices[i+totpar+2];
  cout<<endl;
}


__host__ void GooPdf::setFitControl (FitControl* const fc, bool takeOwnerShip) {
#ifdef CUDADEBUG
  cout<<getName()<<" set the fit control "<<fc<<" : binned? "<<fc->binnedFit()<<" binErr? "<<fc->binErrors()<<endl;
#endif
  for (unsigned int i = 0; i < components.size(); ++i) {
    components[i]->setFitControl(fc, false); 
  }

  if ((fitControl) && (fitControl->getOwner() == this)) {
    delete fitControl; 
    fitControl = 0LL;
  }
  fitControl = fc; 
  if (takeOwnerShip) {
    fitControl->setOwner(this); 
  }
  setMetrics();
}
MetricTaker::MetricTaker (PdfBase* dat, void* dev_functionPtr) 
  : metricIndex(0)
  , functionIdx(dat->getFunctionIndex())
  , parameters(dat->getParameterIndex())
{
  //std::cout << "MetricTaker constructor with " << functionIdx << std::endl; 

  map<void*, int>::iterator localPos = functionAddressToDeviceIndexMap.find(dev_functionPtr); 
  if (localPos != functionAddressToDeviceIndexMap.end()) {
    metricIndex = (*localPos).second; 
  }
  else {
    metricIndex = num_device_functions; 
    host_function_table[num_device_functions] = dev_functionPtr;
    functionAddressToDeviceIndexMap[dev_functionPtr] = num_device_functions; 
    num_device_functions++; 
    MEMCPY_TO_SYMBOL(device_function_table, host_function_table, num_device_functions*sizeof(void*), 0, cudaMemcpyHostToDevice); 
  }
}

BinnedMetricTaker::BinnedMetricTaker (PdfBase* dat, void* dev_functionPtr) 
  : metricIndex(0)
  , functionIdx(dat->getFunctionIndex())
  , parameters(dat->getParameterIndex())
{
  map<void*, int>::iterator localPos = functionAddressToDeviceIndexMap.find(dev_functionPtr); 
  if (localPos != functionAddressToDeviceIndexMap.end()) {
    metricIndex = (*localPos).second; 
  }
  else {
    metricIndex = num_device_functions; 
    host_function_table[num_device_functions] = dev_functionPtr;
    functionAddressToDeviceIndexMap[dev_functionPtr] = num_device_functions; 
    num_device_functions++; 
    MEMCPY_TO_SYMBOL(device_function_table, host_function_table, num_device_functions*sizeof(void*), 0, cudaMemcpyHostToDevice); 
  }
}
//#include "PdfBase.cu" 
