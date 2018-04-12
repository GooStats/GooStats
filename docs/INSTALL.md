# Installation guide for GooStats

1. First, you should install the `GooFit` shipped with this package. I have 
done some modification and it's not compatible with the original [GooFit 
project](https://github.com/GooFit/GooFit)

		# suppose you create a folder GooStats-release
		cd GooStats-release
		git clone git@github.com:DingXuefeng/GooStats.git
		mkdir build_GooFit
		cd build_GooFit
		cmake ../GooStats/GooFit -DCMAKE_INSTALL_PREFIX=../GooFit-install
		make -j4
		make -j4 install
		export GOOFIT_DIR=$(readlink -f ../GooFit-install)
  		# you can put this sentence to your .bashrc
  		echo "export GOOFIT_DIR=$(readlink -f ../GooFit-install)" >> ~/.bashrc
Tips: 
  - for sure remember to compile it on a machine with GPU. if you work on a cluster, use qsub -I 
  - I have turned off OpenMP in GooFit. GooFit can run on GPU, OpenMP and MPI mode.
  However GooStats is only optimized for the GPU mode. Sometimes you work on a cluster with one GPU
  and only one CPU, if you turn on OpenMP, the code will be super slow. I might solve it in the future,
  currently I just turn off the OpenMP in GooFit.

2. Then, try to compile the test project <naive-Reactor>

		mkdir ../build_GooStats
		cd ../GooStats/Modules/naive-Reactor
		#I have writte a Makefile so you can just type make
		make
		./reactor IBD.cfg
3. Start write your own project, see the development guide.
