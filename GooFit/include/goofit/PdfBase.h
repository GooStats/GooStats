#ifndef PDF_BASE_HH
#define PDF_BASE_HH

#include "goofit/Variable.h"
#include "goofit/GlobalCudaDefines.h"
#include "goofit/FitControl.h"
#include <set>
#include "goofit/BinnedDataSet.h"
#include "goofit/UnbinnedDataSet.h"

#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_vector.h>
#include <algorithm>

typedef thrust::counting_iterator<int> IndexIterator;
typedef thrust::constant_iterator<fptype*> DataIterator;
typedef thrust::constant_iterator<int> SizeIterator;
typedef thrust::tuple<IndexIterator, DataIterator, SizeIterator> EventTuple;
typedef thrust::zip_iterator<EventTuple> EventIterator;

const int maxDataSet = 500;
const int maxParams = 500;
const int maxConsts = 4000;
const int maxIndicies = 8000;
extern fptype* dev_event_array[maxDataSet];
extern fptype host_normalisation[maxIndicies];
extern fptype host_params[maxParams];
extern unsigned int host_indices[maxIndicies];
extern int totalParams;
extern int totalConstants;
extern std::string pdfName[maxIndicies];
template<class T>
class DumperPdf;
class SumPdf;
class SumLikelihoodPdf;
extern MEM_CONSTANT fptype cuda_array[maxParams];           // Holds device-side fit parameters. 
extern MEM_DEVICE unsigned int paramIndices[maxIndicies];  // Holds functor-specific indices into cuda_array. Also overloaded to hold integer constants (ie parameters that cannot vary.) 
extern MEM_DEVICE fptype functorConstants[maxConsts];    // Holds non-integer constants. Notice that first entry is number of events. 
extern MEM_CONSTANT fptype normalisationFactors[maxParams]; 

extern MEM_DEVICE void* device_function_table[200];
extern void* host_function_table[200];
extern unsigned int num_device_functions;

class PdfBase {

  public:
    PdfBase (Variable* x, std::string n);

    enum Specials {ForceSeparateNorm = 1, ForceCommonNorm = 2};

    __host__ virtual double calculateNLL () const = 0;
    __host__ virtual fptype normalise () const = 0;
    __host__ void initialiseIndices (std::vector<unsigned int> pindices);

    typedef std::vector<Variable*> obsCont;
    typedef obsCont::iterator obsIter;
    typedef obsCont::const_iterator obsConstIter;
    typedef std::vector<Variable*> parCont;
    typedef parCont::iterator parIter;
    typedef parCont::const_iterator parConstIter;

    __host__ void addSpecialMask (int m) {specialMask |= m;}
    __host__ void copyParams (const std::vector<double>& pars) const;
    __host__ void copyParams ();
    __host__ void copyNormFactors () const;
    __host__ void generateNormRange ();
    __host__ void setName (const std::string &newname) {name = newname;}
    __host__ std::string getName () const {return name;}
    __host__ virtual void getObservables (obsCont& ret) const;
    __host__ virtual void getParameters (parCont& ret) const;
    __host__ Variable* getParameterByName (std::string n) const;
    __host__ int getSpecialMask () const {return specialMask;}
    __host__ void setData (BinnedDataSet* data);
    __host__ void setData (UnbinnedDataSet* data);
    __host__ void setData (std::vector<std::map<Variable*, fptype> >& data);
    __host__ virtual void setFitControl (FitControl* const fc, bool takeOwnerShip = true) = 0;
    __host__ virtual bool hasAnalyticIntegral () const {return false;}
    __host__ unsigned int getFunctionIndex () const {return functionIdx;}
    __host__ unsigned int getParameterIndex () const {return parameters;}
    __host__ unsigned int registerParameter (Variable* var);
    __host__ unsigned int registerConstants (unsigned int amount);
    __host__ virtual void recursiveSetNormalisation (fptype norm = 1) const;
    __host__ void unregisterParameter (Variable* var);
    /// Register a function for this PDF to use in evalution
    template <typename T>
    void registerFunction(std::string name, const T &function) {
        reflex_name_  = name;
        function_ptr_ = get_device_symbol_address(function);
    }
    __host__ void registerObservable (Variable* obs);
    __host__ void setIntegrationFineness (int i);
    __host__ void printProfileInfo (bool topLevel = true);

    __host__ bool parametersChanged () const;
    __host__ void storeParameters () const;

    __host__ obsIter obsBegin () {return observables.begin();}
    __host__ obsIter obsEnd   () {return observables.end();}
    __host__ obsConstIter obsCBegin () const {return observables.begin();}
    __host__ obsConstIter obsCEnd   () const {return observables.end();}

    __host__ void checkInitStatus (std::vector<std::string>& unInited) const;
    void clearCurrentFit ();

  protected:
    std::string reflex_name_; //< This is the name of the type of the PDF, for reflexion purposes. Must be set or
                              // RecursiveSetIndicies must be overloaded.

    void *function_ptr_{nullptr}; //< This is the function pointer to set on the device. Must be set or
                                  // RecursiveSetIndicies must be overloaded.
    fptype numEvents;         // Non-integer to allow weighted events
    unsigned int numEntries;  // Eg number of bins - not always the same as number of events, although it can be.
    fptype* normRanges;       // This is specific to functor instead of variable so that MetricTaker::operator needn't use indices.
    unsigned int parameters;  // Stores index, in 'paramIndices', where this functor's information begins.
    unsigned int cIndex;      // Stores location of constants.
    obsCont observables;
    parCont parameterList;
    FitControl* fitControl;
    std::vector<PdfBase*> components;
    int integrationBins;
    int specialMask; // For storing information unique to PDFs, eg "Normalise me separately" for TddpPdf.
    mutable fptype* cachedParams;
    bool properlyInitialised; // Allows checking for required extra steps in, eg, Tddp and Convolution.

    unsigned int functionIdx; // Stores index of device function pointer.
    int pdfId;

  private:
    std::string name;

    __host__ int registerPdf();
    __host__ void recursiveSetIndices ();
    __host__ void setIndices ();
    friend class DumperPdf<SumLikelihoodPdf>;
};


#endif
