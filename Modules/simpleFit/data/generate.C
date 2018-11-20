void generate() {
  gRandom->SetSeed(1);
  TFile *file = TFile::Open("data.root","RECREATE");
  TH1D *h = new TH1D("data","data",100,2,12);
  TF1 *f = new TF1("f","[](double *x,double *par) { return TMath::Gaus(x[0],par[1],par[2])*par[0]+par[3]; }",2,10,4);
  f->SetParameters(100,5,1,10);
  h->FillRandom("f",10000);
  for(int i = 1;i<=100;++i) {
    std::cout<<i<<" "<<h->GetBinContent(i)<<std::endl;
  }
  h->Write();
  file->Close();
  ofstream out;
  out.open("fbkg.txt");
  out<<100<<" "<<2.05<<" "<<0.1<<std::endl;
  for(int i = 1;i<=100;++i) {
    out<<1<<std::endl;
  }
  out.close();
}
