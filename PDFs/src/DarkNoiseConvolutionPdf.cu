/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "DarkNoiseConvolutionPdf.h"
#include "PdfCache.h"
MEM_CONSTANT fptype* dev_raw_dn_histos[100]; // dark noise histograms for different PDF
EXEC_TARGET fptype device_ConvolveDnHisto(fptype* evt, fptype* , unsigned int* indices) {
  const int workSpaceIndex = RO_CACHE(indices[1]); // ok
  const int npe_lo = RO_CACHE(indices[2]); // ok
  const int dn_max = RO_CACHE(indices[3]); // ok
  const int npe_bin = FLOOR(evt[RO_CACHE(indices[2 + RO_CACHE(indices[0])])])-npe_lo;

  fptype ret     = 0; 
  const unsigned int loop_max = dn_max<npe_bin?dn_max:npe_bin;
  for (unsigned int dn = 0; dn<= loop_max; ++dn) {
    const fptype model = RO_CACHE(PdfCache_dev_array[workSpaceIndex][npe_bin-dn]); 
    const fptype resol = RO_CACHE(dev_raw_dn_histos[workSpaceIndex][dn]);
    ret += model*resol;
#ifdef convolution_CHECK
  const fptype npe = evt[RO_CACHE(indices[2 + RO_CACHE(indices[0])])];
    if(npe==85.5)
      printf("x1 %.2lf x2 %.2lf x3 %.2lf npe %.1lf lo %d npeb %d mb %d rb %d dnmax %d M %.14le R %.14le ret %.14le\n",
	  npe_bin-dn+npe_lo+0.5,dn+0.5,npe,npe,npe_lo,npe_bin,npe_bin-dn,dn,dn_max,model,resol,ret);
#endif
  }
  return ret; 
}
MEM_DEVICE device_function_ptr ptr_to_ConvolveDnHisto = device_ConvolveDnHisto;
DarkNoiseConvolutionPdf::DarkNoiseConvolutionPdf(std::string n,Variable *npe,GooPdf *rawPdf_,BinnedDataSet *dn_histo) :
  GooPdf(npe,n),rawPdf(rawPdf_),dev_iConsts(0LL),
  id(PdfCache::get()->registerFunc(rawPdf)),
  modelWorkSpace(PdfCache_dev_vec[id]),
  npe_lo(npe->lowerlimit>dn_histo->getNumBins()?npe->lowerlimit-dn_histo->getNumBins():0),
  npe_hi(npe->upperlimit),dn_max((*(dn_histo->varsBegin()))->upperlimit-1)
{
  assert(npe->upperlimit-npe->lowerlimit==npe->numbins);
  int numVars = dn_histo->numVariables(); 
  if(numVars!=1) 
    abortWithCudaPrintFlush(__FILE__, __LINE__, getName() + " only valid for 1-D histogram", this);
  { Variable *var = *(dn_histo->varsBegin());
    /* dn histo var start from 0 to n_max */
    if(!(var->lowerlimit==0)||!(var->upperlimit==var->numbins))
      abortWithCudaPrintFlush(__FILE__, __LINE__, getName() + ":  the dark noise histogram binning should be (nthres,0,nthres)", this);
  }
  components.push_back(static_cast<PdfBase*>(rawPdf));
  std::vector<unsigned int> pindices;
  pindices.push_back(id/* index of the dn_histo used by this pdf*/);  // 1
  pindices.push_back(npe_lo);
  pindices.push_back(dn_max);
  copyHistogramToDevice(dn_histo);
  setIntegrationConstants();
  GET_FUNCTION_ADDR(ptr_to_ConvolveDnHisto);
  initialise(pindices); 
}
__host__ void DarkNoiseConvolutionPdf::copyHistogramToDevice (BinnedDataSet *dn_histo) {
  thrust::host_vector<fptype> host_histogram; 
  unsigned int numbins = dn_histo->getNumBins(); 
  for (unsigned int i = 0; i < numbins; ++i) {
    fptype curr = dn_histo->getBinContent(i);
    host_histogram.push_back(curr); // warning: you should normalize the histogram yourself.
  }
  DEVICE_VECTOR<fptype>* dev_dn_histo= new DEVICE_VECTOR<fptype>(host_histogram);  
  static fptype* dev_address[1];
  dev_address[0] = thrust::raw_pointer_cast(dev_dn_histo->data());
  MEMCPY_TO_SYMBOL(dev_raw_dn_histos, dev_address, sizeof(fptype*), id*sizeof(fptype*), cudaMemcpyHostToDevice); 
}
__host__ void DarkNoiseConvolutionPdf::setIntegrationConstants() {
  fptype host_startendN[3];
  host_startendN[0] = npe_lo;
  host_startendN[1] = npe_hi;
  host_startendN[2] = npe_hi-npe_lo;
  if(!dev_iConsts) gooMalloc((void**) &dev_iConsts, 3*sizeof(fptype)); 
  MEMCPY(dev_iConsts, host_startendN, 3*sizeof(fptype), cudaMemcpyHostToDevice); 
}
__host__ fptype DarkNoiseConvolutionPdf::normalise () const {
  // First set normalisation factors to one so we can evaluate convolution without getting zeroes
  recursiveSetNormalisation(fptype(1.0)); 
  //  MEMCPY_TO_SYMBOL(normalisationFactors, host_normalisation, totalParams*sizeof(fptype), 0, cudaMemcpyHostToDevice); 

  // Next recalculate functions at each point, in preparation for convolution integral

  if (rawPdf->parametersChanged()) {
    rawPdf->normalise();
    thrust::counting_iterator<int> binIndex(0); 
    thrust::constant_iterator<fptype*> model_startendN(dev_iConsts); // lo, end, N. eg. 0,100,100, first point would be 0.5, 100-th point would be 99.5
    thrust::constant_iterator<int> model_eventSize(1);
    // Calculate rawPdf function at every point in integration space
    BinnedMetricTaker modalor(rawPdf, getMetricPointer("ptr_to_Eval")); 
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(binIndex, model_eventSize, model_startendN)),
        thrust::make_zip_iterator(thrust::make_tuple(binIndex + modelWorkSpace->size(), model_eventSize, model_startendN)),
        modelWorkSpace->begin(), 
        modalor);
    SYNCH(); 
//#ifdef convolution_CHECK
//    thrust::host_vector<fptype> result(*modelWorkSpace);
//    for(unsigned int i = 0;i<result.size();++i) {
//      std::cout<<i<<"-->(M)"<<result[i]<<std::endl;
//    }
//#endif
    rawPdf->storeParameters();
  }
  host_normalisation[parameters] = 1.0;
  return 1; 
}
#ifdef NLL_CHECK
__host__ fptype DarkNoiseConvolutionPdf::sumOfNll( int numVars ) const {
  static thrust::plus<double> cudaPlus;
  thrust::constant_iterator<int> eventSize(numVars); 
  thrust::constant_iterator<fptype*> arrayAddress(dev_event_array[id]); 
  double dummy = 0;
  thrust::counting_iterator<int> eventIndex(0); 
  DEVICE_VECTOR<fptype> results(numEntries);
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
  fptype* host_array = new fptype[numEntries*3];
  MEMCPY(host_array,dev_event_array[id], 3*numEntries*sizeof(fptype), cudaMemcpyDeviceToHost);
  for(unsigned int i = 0;i<logL.size();++i) {
    double binVolume = host_array[i*3+2];
    sum+= logL[i];
    printf("log(L) %.15le b %lf M %lf tot %.15lf\n",sum,host_array[i*3],host_array[i*3+1],sumV[i]*binVolume);
  }
  if(first) { sum = 0; first = false; } else first = true;
  delete host_array;
  // debug end
  return thrust::reduce(results.begin(),results.end(),dummy,cudaPlus);
}
#endif
