#!/bin/bash

#./reactor data_nuE.cfg plotEnu inputSpectraFiles=data/data_hist.root Data_histName=Enu_hist 2>&1 | tee log
#./reactor data_nuE.cfg plotEnuP inputSpectraFiles=data/data_hist.root Data_histName=Enu_hist_poisson 2>&1 | tee log
#./reactor data_nuE.cfg plotEnuPP inputSpectraFiles=data/data_hist.root Data_histName=Enu_hist_poissonApp 2>&1 | tee log
#./reactor data_recE.cfg plotEv inputSpectraFiles=data/data_hist.root Data_histName=Evis_hist 2>&1 | tee log
#./reactor data_recE.cfg plotEvS inputSpectraFiles=data/data_hist.root Data_histName=Evis_hist_smear 2>&1 | tee log
#./reactor data_recE.cfg plotEvP inputSpectraFiles=data/data_hist.root Data_histName=Evis_hist_poisson 2>&1 | tee log
#./reactor data_recE.cfg plotEvPP inputSpectraFiles=data/data_hist.root Data_histName=Evis_hist_poissonApp 2>&1 | tee log
./reactor data_recE.cfg plotEvPPS inputSpectraFiles=data/data_hist.root Data_histName=Evis_hist_poissonAppSum 2>&1 | tee log
