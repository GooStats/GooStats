> Installation guide for the modified GooFit shipped with GooStats
>
>   Xuefeng Ding http://dingxf.cn
>   Gran Sasso Science Institute
>
>   Copyright (c) 20018
>
> This README is licensed under the GPL v3 License**

1. Install GooFit.

go to some folder, say bx-GooStats/..

	# first login to node with GPU / qsub -I to GPU queue !!
	mkdir build_GooFit
	cd build_GooFit
	cmake ../bx-GooStats/GooFit -DCMAKE_INSTALL_PREFIX="../GooFit-install"
	make -j
	make -j install

that's it.

2. Test the installation

		cd ../
		mkdir build_GooFitTest
		cd build_GooFitTest
		export GOOFIT_DIR="$(pwd)/../GooFit-install"
		cmake ../bx-GooStats/GooFit/test -DCMAKE_MODULE_PATH="$(pwd)/../bx-GooStats/cmake"
		make -j
		./exp

This is a binned likelihood fitting. The fit function is `[0]*exp([1]*x)`.
The fit will be perfomed twice, the first is with ROOT/CPU and the second is
with GOOFIT/GPU, and the result should be  `[0] ~ 500 , [1] ~ -0.5`

If you failed, post an issue on [github/DingXuefeng/GooStats](https://github.com/DingXuefeng/GooStats)
