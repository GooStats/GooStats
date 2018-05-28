/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "SumPdf.h"
#include <utility>
std::map<PdfBase*,int> SumPdf::funMap;
MEM_DEVICE fptype *dev_componentWorkSpace[NPDFSIZE_SumPdf];
DEVICE_VECTOR<fptype>* componentWorkSpace[NPDFSIZE_SumPdf];
MEM_CONSTANT fptype* dev_raw_masks[20];
int SumPdf::maskId = 0;
std::map<BinnedDataSet*,int> SumPdf::maskmap;

EXEC_TARGET fptype device_SumPdfsExt_withSys (fptype* evt, fptype* p, unsigned int* indices) { 
  const int cIndex = RO_CACHE(indices[1]); 
  const fptype npe_val = evt[RO_CACHE(indices[2 + RO_CACHE(indices[0])])]; 
  const fptype npe_lo = RO_CACHE(functorConstants[cIndex]);
  const fptype npe_step = RO_CACHE(functorConstants[cIndex+1]);
  const int npe_bin = (int) FLOOR((npe_val-npe_lo)/npe_step); // no problem with FLOOR: start from 0.5, which corresponse to bin=0
  const int Ncomps = RO_CACHE(indices[2]); 
  fptype ret = 0;
  for (int par = 0; par < Ncomps; ++par) {
    const int workSpaceIndex = RO_CACHE(indices[par+3]);
    const fptype weight = RO_CACHE(p[RO_CACHE(indices[par+3+Ncomps])]);
    const int i = RO_CACHE(indices[par+3+Ncomps+Ncomps]);
    const fptype sysi = i?RO_CACHE(p[i]):1;
    const fptype curr = dev_componentWorkSpace[workSpaceIndex][npe_bin];
    ret += weight * sysi * curr; // normalization is always 1
#ifdef convolution_CHECK
    if(npe_bin==0)
      //  if(THREADIDX==10)
      printf("+ npe %.1lf npebin %d ret %.10lf w %.10lf wk %d cur %.10lf sysi %d sys %.10lf\n",npe_val,npe_bin,ret,weight,workSpaceIndex,curr,i,sysi);
#endif
  }
  const int i = RO_CACHE(indices[Ncomps+3+Ncomps+Ncomps]);
  const fptype sys = i?RO_CACHE(p[i]):1;
  ret *= RO_CACHE(functorConstants[cIndex+2]) * sys; // exposure
#ifdef convolution_CHECK
    if(npe_bin==0)
      //  if(THREADIDX==10)
      printf("+ npe %.1lf npebin %d ret %.10lf sysi %d sys %.10lf\n",npe_val,npe_bin,ret,i,sys);
#endif
  //  printf("npe %.1lf ret %.10lf\n",npe_val,ret);

  return ret; 
}

EXEC_TARGET fptype device_SumPdfsExt (fptype* evt, fptype* p, unsigned int* indices) { 
  const int cIndex = RO_CACHE(indices[1]); 
  const fptype npe_val = evt[RO_CACHE(indices[2 + RO_CACHE(indices[0])])]; 
  const fptype npe_lo = RO_CACHE(functorConstants[cIndex]);
  const fptype npe_step = RO_CACHE(functorConstants[cIndex+1]);
  const int npe_bin = (int) FLOOR((npe_val-npe_lo)/npe_step); // no problem with FLOOR: start from 0.5, which corresponse to bin=0
  const int Ncomps = RO_CACHE(indices[2]); 
  fptype ret = 0;
  for (int par = 0; par < Ncomps; ++par) {
    const int workSpaceIndex = RO_CACHE(indices[par+3]);
    const fptype weight = RO_CACHE(p[RO_CACHE(indices[par+3+Ncomps])]);
    const fptype curr = dev_componentWorkSpace[workSpaceIndex][npe_bin];
    ret += weight * curr; // normalization is always 1
#ifdef convolution_CHECK
    if(npe_bin==0)
      //  if(THREADIDX==10)
      printf("+ npe %.1lf npebin %d ret %.10lf w %.10lf wk %d cur %.10lf\n",npe_val,npe_bin,ret,weight,workSpaceIndex,curr);
#endif
  }
  ret *= RO_CACHE(functorConstants[cIndex+2]); // exposure
  //  printf("npe %.1lf ret %.10lf\n",npe_val,ret);

  return ret; 
}
EXEC_TARGET fptype device_SumPdfsExtMask (fptype* evt, fptype* p, unsigned int* indices) { 
  const int cIndex = RO_CACHE(indices[1]); 
  const fptype npe_val = evt[RO_CACHE(indices[2 + RO_CACHE(indices[0])])]; 
  const fptype npe_lo = RO_CACHE(functorConstants[cIndex]);
  const fptype npe_step = RO_CACHE(functorConstants[cIndex+1]);
  const int npe_bin = (int) FLOOR((npe_val-npe_lo)/npe_step); // no problem with FLOOR: start from 0.5, which corresponse to bin=0
  const int Ncomps = RO_CACHE(indices[2]); 
  fptype ret = 0;
#ifdef convolution_CHECK
  if(npe_bin==0)
    printf("cIndex %d npe %.5le npelo %.5le npestep %.5le npebin %d Ncomps %d \n",cIndex,npe_val,npe_lo,npe_step,npe_bin,Ncomps);
#endif
  for (int par = 0; par < Ncomps; ++par) {
    const int workSpaceIndex = RO_CACHE(indices[par+3]);
    const int maskSpaceIndex = RO_CACHE(indices[par+3+Ncomps]);
    const fptype curr = dev_componentWorkSpace[workSpaceIndex][npe_bin];
    const fptype mask = RO_CACHE(dev_raw_masks[maskSpaceIndex][npe_bin]);
    ret += curr*mask; // normalization is always 1
#ifdef Mask_CHECK
    //  if(THREADIDX==0)
    printf("npe %.1lf npebin %d wid %d mid %d w %.14le m %.14le curr %.14le ret %.14le\n",npe_val,npe_bin,workSpaceIndex,maskSpaceIndex,curr,mask,curr*mask,ret);
#endif
  }
  // ret *= RO_CACHE(functorConstants[cIndex+2]); // exposure

  return ret; 
}
EXEC_TARGET fptype device_SumPdfsExtSimple (fptype* evt, fptype* p, unsigned int* indices) { 
  const int cIndex = RO_CACHE(indices[1]); 
  const fptype npe_val = evt[RO_CACHE(indices[2 + RO_CACHE(indices[0])])]; 
  const fptype npe_lo = RO_CACHE(functorConstants[cIndex]);
  const fptype npe_step = RO_CACHE(functorConstants[cIndex+1]);
  const int npe_bin = (int) FLOOR((npe_val-npe_lo)/npe_step); // no problem with FLOOR: start from 0.5, which corresponse to bin=0
  const int Ncomps = RO_CACHE(indices[2]); 
  fptype ret = 0;
  for (int par = 0; par < Ncomps; ++par) {
    const int workSpaceIndex = RO_CACHE(indices[par+3]);
    const fptype curr = dev_componentWorkSpace[workSpaceIndex][npe_bin];
    ret += curr; // normalization is always 1
  }
  //  ret *= RO_CACHE(functorConstants[cIndex+2]); // exposure

  return ret; 
}


MEM_DEVICE device_function_ptr ptr_to_SumPdfsExt_withSys = device_SumPdfsExt_withSys; 
MEM_DEVICE device_function_ptr ptr_to_SumPdfsExt = device_SumPdfsExt; 
MEM_DEVICE device_function_ptr ptr_to_SumPdfsExtMask = device_SumPdfsExtMask; 
MEM_DEVICE device_function_ptr ptr_to_SumPdfsExtSimple = device_SumPdfsExtSimple; 

  SumPdf::SumPdf (std::string n, const fptype norm_,const std::vector<Variable*> &weights, const std::vector<Variable*> &sysi,Variable *sys,const std::vector<PdfBase*> &comps,Variable *npe)
  : GooPdf(npe, n) 
  , norm(norm_)
    , extended(true), _weights(weights),dataset(nullptr)
{
  assert(weights.size() == comps.size());
  assert(weights.size() == sysi.size());
  assert(norm>0);
  set_startstep(norm);
  register_components(comps,npe->numbins);

  for (unsigned int w = 0; w < weights.size(); ++w) {
    assert(weights.at(w));
    pindices.push_back(registerParameter(weights[w])); 
  }

  for (unsigned int w = 0; w < sysi.size(); ++w) {
    if(sysi.at(w)) 
      pindices.push_back(registerParameter(sysi[w])); 
    else
      pindices.push_back(0);
  }

  if(sys)
    pindices.push_back(registerParameter(sys));
  else
    pindices.push_back(0);

  GET_FUNCTION_ADDR(ptr_to_SumPdfsExt_withSys);

  initialise(pindices); 
} 

  SumPdf::SumPdf (std::string n, const fptype norm_,const std::vector<Variable*> &weights, const std::vector<PdfBase*> &comps,Variable *npe)
  : GooPdf(npe, n) 
  , norm(norm_)
    , extended(true), _weights(weights),dataset(nullptr)
{
  assert(weights.size() == comps.size());
  assert(norm>0);
  set_startstep(norm);
  register_components(comps,npe->numbins);

  for (unsigned int w = 0; w < weights.size(); ++w) {
    assert(weights.at(w));
    pindices.push_back(registerParameter(weights[w])); 
  }

  GET_FUNCTION_ADDR(ptr_to_SumPdfsExt);

  initialise(pindices); 
} 
  SumPdf::SumPdf (std::string n, const std::vector<PdfBase*> &comps,const std::vector<const BinnedDataSet*> &mask,Variable *npe)
  : GooPdf(npe, n) 
    , extended(true),_weights(),dataset(nullptr)
{
  set_startstep(0);
  register_components(comps,npe->numbins);
  copyHistogramToDevice(mask);
  GET_FUNCTION_ADDR(ptr_to_SumPdfsExtMask);
  initialise(pindices); 
} 
  SumPdf::SumPdf (std::string n, const std::vector<PdfBase*> &comps,Variable *npe)
  : GooPdf(npe, n) 
    , extended(true),dataset(nullptr)
{
  set_startstep(0);
  register_components(comps,npe->numbins);
  GET_FUNCTION_ADDR(ptr_to_SumPdfsExtSimple);
  initialise(pindices); 
}
void SumPdf::set_startstep(fptype norm) {
  Variable *npe = *(observables.begin());
  pindices.push_back(registerConstants(3));
  fptype host_iConsts[3];
  host_iConsts[0] = npe->lowerlimit;
  host_iConsts[1] = (npe->upperlimit-npe->lowerlimit)/npe->numbins;
  host_iConsts[2] = norm;
  MEMCPY_TO_SYMBOL(functorConstants, host_iConsts, 3*sizeof(fptype), cIndex*sizeof(fptype), cudaMemcpyHostToDevice);  // cIndex is a member derived from PdfBase and is set inside registerConstants method

  gooMalloc((void**) &dev_iConsts, 3*sizeof(fptype)); 
  host_iConsts[0] = npe->lowerlimit;
  host_iConsts[1] = npe->upperlimit;
  host_iConsts[2] = npe->numbins;
  MEMCPY(dev_iConsts, host_iConsts, 3*sizeof(fptype), cudaMemcpyHostToDevice); 
}
void SumPdf::register_components(const std::vector<PdfBase*> &comps,int N) {
  components = comps;
  pindices.push_back(components.size());
  for (unsigned int w = 0; w < components.size(); ++w) {
    PdfBase *pdf = components.at(w);
    assert(pdf);
    if(funMap.find( pdf ) == funMap.end()) {
      const int workSpaceIndex = registerFunc(components.at(w));
      assert(workSpaceIndex<NPDFSIZE_SumPdf);
      componentWorkSpace[workSpaceIndex] = new DEVICE_VECTOR<fptype>(N);
      fptype *dev_address = thrust::raw_pointer_cast(componentWorkSpace[workSpaceIndex]->data());
      MEMCPY_TO_SYMBOL(dev_componentWorkSpace, &dev_address, sizeof(fptype*), workSpaceIndex*sizeof(fptype*), cudaMemcpyHostToDevice); 
      //MEMCPY(dev_componentWorkSpace,&dev_address,sizeof(fptype*),cudaMemcpyHostToDevice);
    } 
    pindices.push_back(funMap.at(pdf));
  }
}

__host__ fptype SumPdf::normalise () const {
  host_normalisation[parameters] = 1.0; 
  thrust::counting_iterator<int> binIndex(0); 

  Variable *npe = *(observables.begin());
  m_updated = false;
  for (unsigned int i = 0; i < components.size(); ++i) {
    GooPdf *component = dynamic_cast<GooPdf*>(components.at(i));
    if (component->parametersChanged()) {
      m_updated = true;
      component->normalise();  // this is needed for the GeneralConvolution
      thrust::constant_iterator<fptype*> startendstep(dev_iConsts); // 3*fptype lo, hi and step for npe
      thrust::constant_iterator<int> eventSize(1); // 1: only npe
      BinnedMetricTaker modalor(component, getMetricPointer("ptr_to_Eval")); 
      const int workSpaceIndex = funMap.at(components.at(i));
      thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(binIndex, eventSize, startendstep)),
	  thrust::make_zip_iterator(thrust::make_tuple(binIndex + npe->numbins, eventSize, startendstep)),
	  componentWorkSpace[workSpaceIndex]->begin(),
	  modalor);
      SYNCH(); 
      component->storeParameters();
    }
  }
  return 1; 
}

int SumPdf::registerFunc(PdfBase *pdf) {
  static int pdf_Id = -1;
  if(funMap.find( pdf ) == funMap.end()) {
    funMap.insert(std::make_pair(pdf,++pdf_Id));
    printf("SumPdf::registerFunc register [%s](%p) as [%d]\n",
	pdf->getName().c_str(),pdf,funMap.at(pdf));
  }
  return funMap.at(pdf);
}
int SumPdf::registerMask(BinnedDataSet *mask) {
  if(maskmap.find(mask)==maskmap.end()) {
    maskmap.insert(std::make_pair(mask,maskId++));
  }
  return maskmap.at(mask);
}
void SumPdf::copyHistogramToDevice(const std::vector<const BinnedDataSet*> &masks) {
  for(auto maskIt = masks.begin();maskIt!=masks.end();++maskIt) {
    int maskId_ = registerMask(const_cast<BinnedDataSet*>(*maskIt));
    copyHistogramToDevice(*maskIt,maskId_);
    pindices.push_back(maskId_);
  }
}
#include <cassert>
void SumPdf::copyHistogramToDevice(const BinnedDataSet* mask,int id) {
  thrust::host_vector<fptype> host_histogram; 
  Variable *mask_npe = *(mask->varsBegin());
  Variable *npe = *(observables.begin());
  int shift = npe->lowerlimit-mask_npe->lowerlimit;
  for (int i = 0; i < npe->numbins; ++i) {
    fptype curr = mask->getBinContent(i+shift);
    host_histogram.push_back(curr); // warning: you should normalize the histogram yourself.
  }
  DEVICE_VECTOR<fptype>* dev_mask= new DEVICE_VECTOR<fptype>(host_histogram);  
  static fptype* dev_address[1];
  dev_address[0] = thrust::raw_pointer_cast(dev_mask->data());
  assert(id<NPDFSIZE_SumPdf);
  MEMCPY_TO_SYMBOL(dev_raw_masks, dev_address, sizeof(fptype*), id*sizeof(fptype*), cudaMemcpyHostToDevice); 
#ifdef convolution_CHECK
  fptype *raw_dn_addr[1];
  MEMCPY_FROM_SYMBOL(raw_dn_addr,dev_raw_masks,sizeof(fptype*),id*sizeof(fptype*),cudaMemcpyDeviceToHost);
  int numbins = npe->numbins;
  fptype *host_array = new fptype[numbins];
  MEMCPY(host_array,raw_dn_addr[0],numbins*sizeof(fptype),cudaMemcpyDeviceToHost);
  for(int i = 0;i<numbins;++i) {
    std::cout<<id<<" : "<<i+shift<<"-->"<<host_array[i]<<std::endl;
  }
  delete [] host_array;
#endif
}
#ifdef NLL_CHECK
#include "GooStatsNLLCheck.h"
__host__ double SumPdf::sumOfNll (int numVars) const {
  static thrust::plus<fptype> cudaPlus;
  thrust::constant_iterator<int> eventSize(numVars); 
  thrust::constant_iterator<fptype*> arrayAddress(dev_event_array[pdfId]); 
  fptype dummy = 0;
  thrust::counting_iterator<int> eventIndex(0); 
  DEVICE_VECTOR<fptype> results(numEntries);
  assert(numEntries);
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(eventIndex, arrayAddress, eventSize)),
      thrust::make_zip_iterator(thrust::make_tuple(eventIndex + numEntries, arrayAddress, eventSize)),
      results.begin(),
      *logger);
  // debug start
  static double sum = 0;
  static bool first = false;
  thrust::host_vector<fptype> logL = results; 
  DEVICE_VECTOR<fptype> dev_sumV(numEntries);
  MetricTaker modalor((PdfBase*)(this), getMetricPointer("ptr_to_Eval")); 
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(eventIndex, arrayAddress, eventSize)),
      thrust::make_zip_iterator(thrust::make_tuple(eventIndex + numEntries, arrayAddress, eventSize)),
      dev_sumV.begin(),
      modalor);
  thrust::host_vector<fptype> sumV = dev_sumV;
  thrust::host_vector<fptype> fVal[200];
  for(unsigned int j = 0;j<components.size();++j) {
    const int workSpaceIndex = funMap.at(components.at(j));
    fVal[workSpaceIndex] = *(componentWorkSpace[workSpaceIndex]);
  }
  fptype* host_array = new fptype[numEntries*3];
  MEMCPY(host_array,dev_event_array[pdfId], 3*numEntries*sizeof(fptype), cudaMemcpyDeviceToHost);
  for(unsigned int i = 0;i<logL.size();++i) {
    double binVolume = host_array[i*3+2];
    sum+= logL[i];
//    printf("log(L) %.12le b %lf M %lf tot %.12le\n",sum,host_array[i*3],host_array[i*3+1],sumV[i]*binVolume);
    GooStatsNLLCheck::get()->record_LL(i,host_array[i*3],host_array[i*3+1],sumV[i]*binVolume,logL[i]);
    for(unsigned int j = 0;j<components.size();++j) {
      const int workSpaceIndex = funMap.at(components.at(j));
      double result = fVal[workSpaceIndex][i]*norm*binVolume*host_params[host_indices[parameters+3+j+components.size()]];
//      printf(" %s %.12le",components.at(j)->getName().c_str(),result);
      GooStatsNLLCheck::get()->record_species(i,components.at(j)->getName(),result);
    }
//    printf("\n");
  }
  if(first) { sum = 0; first = false; } else first = true;
  delete host_array;
  // debug end
  return thrust::reduce(results.begin(),results.end(),dummy,cudaPlus);
}
#endif
#include "TRandom.h"
std::unique_ptr<fptype []> SumPdf::fill_random() {
  copyParams(); 
  normalise();
  int dimensions = 2 + observables.size(); // Bin center (x,y, ...), bin value, and bin volume.

  if(parametersChanged()) {
    DEVICE_VECTOR<fptype> dev_sumV(numEntries);
    MetricTaker modalor((PdfBase*)(this), getMetricPointer("ptr_to_Eval")); 
    thrust::constant_iterator<int> eventSize(-(observables.size()+2));
    thrust::constant_iterator<fptype*> arrayAddress(dev_event_array[pdfId]); 
    thrust::counting_iterator<int> eventIndex(0); 
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(eventIndex, arrayAddress, eventSize)),
	thrust::make_zip_iterator(thrust::make_tuple(eventIndex + numEntries, arrayAddress, eventSize)),
	dev_sumV.begin(),
	modalor);
    cached_sumV = dev_sumV;
  }

  std::unique_ptr<fptype[]> h_ptr(new fptype[numEntries*dimensions]);
  fptype* host_array = h_ptr.get();
  MEMCPY(host_array,dev_event_array[pdfId], dimensions*numEntries*sizeof(fptype), cudaMemcpyDeviceToHost);
  numEvents = 0;
  for(unsigned int i = 0; i < numEntries; ++i) {
    fptype new_value = gRandom->Poisson(cached_sumV[i]*host_array[i*dimensions+observables.size()+1]);
    //    std::cout<<"("<<i<<") before ["<<host_array[i*dimensions + observables.size()+0]<<"] after ["<<new_value<<"] ("<<cached_sumV[i]<<","<<host_array[i*dimensions+observables.size()+1]<<")"<<std::endl;
    host_array[i*dimensions + observables.size()+0] = new_value;
    numEvents += host_array[i*dimensions + observables.size()+0];
  }

  if(dev_event_array[pdfId]) {
    gooFree(dev_event_array[pdfId]);
    dev_event_array[pdfId] = 0;
  }
  gooMalloc((void**) &(dev_event_array[pdfId]), dimensions*numEntries*sizeof(fptype));
  MEMCPY(dev_event_array[pdfId], host_array, dimensions*numEntries*sizeof(fptype), cudaMemcpyHostToDevice);
  MEMCPY_TO_SYMBOL(functorConstants, &numEvents, sizeof(fptype), 0, cudaMemcpyHostToDevice);
  return h_ptr;
}
BinnedDataSet *SumPdf::getData() {
  if(dataset) return dataset; else {
    copyParams(); 
    normalise();
    int dimensions = 2 + observables.size(); // Bin center (x,y, ...), bin value, and bin volume.

    std::unique_ptr<fptype[]> h_ptr(new fptype[numEntries*dimensions]);
    fptype* host_array = h_ptr.get();
    MEMCPY(host_array,dev_event_array[pdfId], dimensions*numEntries*sizeof(fptype), cudaMemcpyDeviceToHost);
    // [0] Bin center
    // [1] Nevent(experiment)
    // [2] Bin volume
    Variable *obs = observables.front();
    dataset = new BinnedDataSet(obs);
    for(unsigned int i = 0; i < numEntries; ++i) {
      dataset->setBinContent(i,host_array[i*dimensions+1]);
    }
    return getData();
  }
}
double SumPdf::Chi2() {
  setFitControl(new BinnedChisqFit());
  copyParams();
  return calculateNLL();
}
int SumPdf::NDF() {
  int NnonZeroBins = 0; {
    getData();
    for(unsigned int i = 0; i < numEntries; ++i) {
      if(dataset->getBinContent(i)>0) NnonZeroBins++;
    }
  }
  int NfreePar = 0; {
    parCont params;
    getParameters(params); 
    std::set<Variable*> pars;
    for(auto par: params) {
      pars.insert(par);
    }
    for(auto par: pars) {
      if(!(par->fixed || par->error == 0)) NfreePar++;
    }
  }
  return NnonZeroBins - NfreePar;
}
