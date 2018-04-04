/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "MultiVariatePdf.h"
int MultiVariatePdf::totalPdf = 0;
MEM_CONSTANT fptype* dev_mv_k[100];
MEM_CONSTANT fptype* dev_mv_n0[100];
MEM_CONSTANT fptype* dev_mv_n1[100];
DEVICE_VECTOR<fptype> *dev_vec_mv_k[100];
DEVICE_VECTOR<fptype> *dev_vec_mv_n0[100];
DEVICE_VECTOR<fptype> *dev_vec_mv_n1[100];
MEM_CONSTANT fptype dev_mv_m0[100];
MEM_CONSTANT fptype dev_mv_m1[100];
template<MultiVariatePdf::MVLLType T>
EXEC_TARGET fptype MVLL(const fptype k,const fptype n0,const fptype n1,const fptype m0,const fptype m1);
#include "MultiVariatePdf.icc" // concrete implementation of the MultiVariate Likelihood
template<MultiVariatePdf::MVLLType T>
EXEC_TARGET fptype device_MV(fptype* evt, fptype* p, unsigned int* indices) {
  const fptype mv_val = evt[RO_CACHE(indices[2 + RO_CACHE(indices[0])])]; 
  const int cIndex = RO_CACHE(indices[1]);
  const fptype mv_lo = RO_CACHE(functorConstants[cIndex]);
  const fptype mv_step = RO_CACHE(functorConstants[cIndex+1]);
  const int mv_bin = (int) FLOOR((mv_val-mv_lo)/mv_step); // no problem with FLOOR: start from 0.5, which corresponse to bin=0
  const int MVid = RO_CACHE(indices[2]);
  const fptype m0 = RO_CACHE(dev_mv_m0[MVid]);
  const fptype m1 = RO_CACHE(dev_mv_m1[MVid]);
  const fptype k = RO_CACHE(dev_mv_k[MVid][mv_bin]);
  const fptype n0 = RO_CACHE(dev_mv_n0[MVid][mv_bin]);
  const fptype n1 = RO_CACHE(dev_mv_n1[MVid][mv_bin]);

  const fptype ret = MVLL<T>(k,n0,n1,m0,m1);
//#ifdef NLL_CHECK
//  printf("k %lf n0 %lf n1 %lf m0 %lf m1 %lf ret %le\n",
//      fptype(k),fptype(n0),fptype(n1),m0,m1,EXP(ret));
//#endif
  
  return -ret; 
}
MEM_DEVICE device_function_ptr ptr_to_MV_StefanoDavini = device_MV<MultiVariatePdf::MVLLType::StefanoDavini>; 
#include "SumPdf.h"
const std::vector<int> MultiVariatePdf::get_pdfids(const std::vector<GooPdf*> &pdfs) {
  std::vector<int> ids;
  for(auto pdf : pdfs) 
    ids.push_back(SumPdf::registerFunc(static_cast<PdfBase*>(pdf)));
  return ids;
}
const std::vector<int> MultiVariatePdf::get_Nids(const std::vector<Variable*> &rates) {
  std::vector<int> ids;
  for(auto rate : rates) 
    ids.push_back(registerParameter(rate));
  return ids;
}
MultiVariatePdf::MultiVariatePdf(std::string n, MVLLType MVLLtype,Variable *mv_var,BinnedDataSet *data,const std::vector<BinnedDataSet*> &refs,const std::vector<GooPdf*> &pdf_0_,const std::vector<GooPdf*> &pdf_1_,const std::vector<Variable*> &rate_0_,const std::vector<Variable*> &rate_1_,int startbin_,int endbin_/*startbin<=bin<endbin*/,SumPdf *sumpdf_,double binVolume_ ) :
  GooPdf(mv_var,n),
  pdf_0(get_pdfids(pdf_0_)),
  pdf_1(get_pdfids(pdf_1_)),
  rate_0(get_Nids(rate_0_)),
  rate_1(get_Nids(rate_1_)),
  binVolume(binVolume_),
  sumpdf(sumpdf_),
  Nbin(data->getNumBins()),
  MVid(totalPdf++),
  startbin(startbin_-static_cast<int>((*(sumpdf->obsCBegin()))->lowerlimit)),
  endbin(endbin_-static_cast<int>((*(sumpdf->obsCBegin()))->lowerlimit)),
  dev_iConsts(0LL)
{
  copyTH1DToGPU(data,refs);
  std::vector<unsigned int> pindices;
  pindices.push_back(registerConstants(2));
  pindices.push_back(MVid/* index of the dn_histo used by this pdf*/);  // 1
  switch(MVLLtype) {
    case MVLLType::StefanoDavini:
      GET_FUNCTION_ADDR(ptr_to_MV_StefanoDavini);
      break;
    default:
      abortWithCudaPrintFlush(__FILE__, __LINE__, getName() + " unknown MVLLtype", this);
  }
  initialise(pindices); 

  gooMalloc((void**) &dev_iConsts, 3*sizeof(fptype)); 
  fptype host_iConsts[3];
  host_iConsts[0] = mv_var->lowerlimit;
  host_iConsts[1] = (mv_var->upperlimit-mv_var->lowerlimit)/mv_var->numbins;
  MEMCPY_TO_SYMBOL(functorConstants, host_iConsts, 2*sizeof(fptype), cIndex*sizeof(fptype), cudaMemcpyHostToDevice);  // cIndex is a member derived from PdfBase and is set inside registerConstants method
  host_iConsts[1] = mv_var->upperlimit;
  host_iConsts[2] = mv_var->numbins;
  MEMCPY(dev_iConsts, host_iConsts, 3*sizeof(fptype), cudaMemcpyHostToDevice); 
}
void MultiVariatePdf::copyTH1DToGPU(BinnedDataSet *data,const std::vector<BinnedDataSet*> &refs) {
  fptype* dev_address[1];
  copyTH1DToGPU(data,sum_k,dev_address,dev_vec_mv_k[MVid]);
  MEMCPY_TO_SYMBOL(dev_mv_k,dev_address,sizeof(fptype*),MVid*sizeof(fptype*),cudaMemcpyHostToDevice);
  copyTH1DToGPU(refs.at(0),I0,dev_address,dev_vec_mv_n0[MVid]);
  MEMCPY_TO_SYMBOL(dev_mv_n0,dev_address,sizeof(fptype*),MVid*sizeof(fptype*),cudaMemcpyHostToDevice);
  copyTH1DToGPU(refs.at(1),I1,dev_address,dev_vec_mv_n1[MVid]);
  MEMCPY_TO_SYMBOL(dev_mv_n1,dev_address,sizeof(fptype*),MVid*sizeof(fptype*),cudaMemcpyHostToDevice);
}
void MultiVariatePdf::copyTH1DToGPU(BinnedDataSet *data,fptype &sum,fptype *dev_address[1],DEVICE_VECTOR<fptype> *& dev_vec_address) {
  thrust::host_vector<fptype> host_histogram; 
  unsigned int numbins = data->getNumBins(); 
  sum=0;
  for (unsigned int i = 0; i < numbins; ++i) {
    fptype curr = data->getBinContent(i);
    sum+= curr;
    host_histogram.push_back(curr); // warning: you should normalize the histogram yourself.
  }
  dev_vec_address= new DEVICE_VECTOR<fptype>(host_histogram);  
  dev_address[0] = thrust::raw_pointer_cast(dev_vec_address->data());
}
__host__ fptype MultiVariatePdf::normalise () const {
  sumpdf->normalise();
  host_normalisation[parameters] = 1.0; 
  return 1;
}
__host__ double MultiVariatePdf::calculateNLL () const {
  if(IsChisquareFit()) return 0;
  return GooPdf::calculateNLL();
}
__host__ fptype MultiVariatePdf::sumOfNll (int __attribute__((__unused__)) numVars) const {
  static fptype logL;
  //if (sumpdf->updated()) {
    calculate_m0m1();
    static thrust::plus<fptype> cudaPlus;
    thrust::counting_iterator<int> binIndex(0); 
    thrust::constant_iterator<fptype*> startendstep(dev_iConsts); // 3*fptype lo, hi and step for npe
    thrust::constant_iterator<int> eventSize(1); // 1: only npe
    fptype dummy = 0;
    BinnedMetricTaker modalor(const_cast<MultiVariatePdf*>(this), getMetricPointer("ptr_to_Eval")); 
    logL = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(binIndex, eventSize, startendstep)),
        thrust::make_zip_iterator(thrust::make_tuple(binIndex + Nbin, eventSize, startendstep)),
        modalor,dummy,cudaPlus);
  //}
#ifdef NLL_CHECK
    DEVICE_VECTOR<fptype> dev_logLs(Nbin);
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(binIndex, eventSize, startendstep)),
        thrust::make_zip_iterator(thrust::make_tuple(binIndex + Nbin, eventSize, startendstep)),
        dev_logLs.begin(),
        modalor);
    thrust::host_vector<fptype> logLs(dev_logLs);
    thrust::host_vector<fptype> k(*dev_vec_mv_k[MVid]);
    thrust::host_vector<fptype> n0(*dev_vec_mv_n0[MVid]);
    thrust::host_vector<fptype> n1(*dev_vec_mv_n1[MVid]);
    double sum = 0;
    const Variable *mv_var = *obsCBegin();
    double lo = mv_var->lowerlimit;
    double de = (mv_var->upperlimit-lo)/mv_var->numbins;
    fptype m0,m1;
    MEMCPY_FROM_SYMBOL(&m0,dev_mv_m0,sizeof(fptype),MVid*sizeof(fptype),cudaMemcpyDeviceToHost);
    MEMCPY_FROM_SYMBOL(&m1,dev_mv_m1,sizeof(fptype),MVid*sizeof(fptype),cudaMemcpyDeviceToHost);
    for(unsigned int i = 0;i<logLs.size();++i) {
      sum+=logLs[i];
      printf("log(L)MV %.15le e %lf user %lf k %.2lf n0 %.2lf n1 %.2lf m0 %.15le m1 %.15le L %.15le\n",
          sum,(startbin+endbin)/2.+static_cast<int>((*(sumpdf->obsCBegin()))->lowerlimit),(i+0.5)*de+lo,
          k[i],n0[i],n1[i],m0,m1,logLs[i]);
    }
#endif
  return logL;
}
void MultiVariatePdf::calculate_m0m1() const {
#ifdef SEPARABLE
  extern DEVICE_VECTOR<fptype>* componentWorkSpace[100];
#endif
  fptype N0 = 0,N1 = 0;
  auto rateIdit = rate_0.begin();
  for(auto pdfId : pdf_0) {
    const double scale = sumpdf->Norm()*binVolume*host_params[*rateIdit];
    N0 += thrust::reduce(componentWorkSpace[pdfId]->begin()+startbin,componentWorkSpace[pdfId]->begin()+endbin)*scale;
#ifdef MV_CHECK
    thrust::host_vector<fptype> values(*componentWorkSpace[pdfId]);
    double sum = 0; int i = static_cast<int>((*(sumpdf->obsBegin()))->lowerlimit);
    for(auto value : values) { 
      sum+=value;
      printf("N%d [%d](%d)<%lf>-><%lf> <%lf>\n",0,pdfId,i,i+0.5,value*scale,sum*scale);
      ++i;
    }
    printf("N0 %lf [%d] a(%d)-b(%d)->(%lf) tot 0-%d (%lf)\n",N0,pdfId,
        startbin,endbin,thrust::reduce(componentWorkSpace[pdfId]->begin()+startbin,componentWorkSpace[pdfId]->begin()+endbin)*scale,
        componentWorkSpace[pdfId]->size(),thrust::reduce(componentWorkSpace[pdfId]->begin(),componentWorkSpace[pdfId]->end())*scale);
#endif
    ++rateIdit;
  }
  rateIdit = rate_1.begin();
  for(auto pdfId : pdf_1) {
    const double scale = sumpdf->Norm()*binVolume*host_params[*rateIdit];
    N1 += thrust::reduce(componentWorkSpace[pdfId]->begin()+startbin,componentWorkSpace[pdfId]->begin()+endbin)*scale;
#ifdef MV_CHECK
    thrust::host_vector<fptype> values(*componentWorkSpace[pdfId]);
    double sum = 0; int i = static_cast<int>((*(sumpdf->obsBegin()))->lowerlimit);
    for(auto value : values) { 
      sum+=value;
      printf("N%d [%d](%d)<%lf>-><%lf> <%lf>\n",1,pdfId,i,i+0.5,value*scale,sum*scale);
      ++i;
    }
    printf("N1 %lf [%d] a(%d)-b(%d)->(%lf) tot 0-%d (%lf)\n",N1,pdfId,
        startbin,endbin,thrust::reduce(componentWorkSpace[pdfId]->begin()+startbin,componentWorkSpace[pdfId]->begin()+endbin)*scale,
        componentWorkSpace[pdfId]->size(),thrust::reduce(componentWorkSpace[pdfId]->begin(),componentWorkSpace[pdfId]->end())*scale);
#endif
    ++rateIdit;
  }
  fptype host_m0 = N0/(N0+N1)*sum_k/(I0+Nbin);
  fptype host_m1 = N1/(N0+N1)*sum_k/(I1+Nbin);
#ifdef NLL_CHECK
  printf("n %d i %d Ni %lf Nsum %lf sum_k %.1lf Ii %.1lf Nbin %d\n",2,0,N0,N0+N1,sum_k,I0,Nbin);
  printf("n %d i %d Ni %lf Nsum %lf sum_k %.1lf Ii %.1lf Nbin %d\n",2,1,N1,N0+N1,sum_k,I1,Nbin);
#endif
  MEMCPY_TO_SYMBOL(dev_mv_m0,&host_m0,sizeof(fptype),MVid*sizeof(fptype),cudaMemcpyHostToDevice);
  MEMCPY_TO_SYMBOL(dev_mv_m1,&host_m1,sizeof(fptype),MVid*sizeof(fptype),cudaMemcpyHostToDevice);
}
