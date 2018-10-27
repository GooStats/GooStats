# Installation guide for GooStats

- Download GooStats

	  mkdir GooStats-release
	  cd GooStats-release
	  git clone git@github.com:GooStats/GooStats.git
	  cd GooStats
	  GooStatsVersion=4.0.1
	  git checkout ${GooStatsVersion}
	  cd ../..

- symbol link the `download.sh` and `compile.sh` I created for you, and download GooStats

	  ln -s GooStats-release/GooStats/setup/download.sh
	  ln -s GooStats-release/GooStats/setup/compile.sh
	  ./download.sh
- this script will create a script for setting up environment `setup.sh`, and you might want to inlcude it in `.bashrc`

	  echo "source $(readlink -f GooStats-release/setup.sh)" >> ~/.bashrc
- If you are working on a cluster, launch an interactive job to a cluster node equipped with GPU: `qsub -q gpu -I`
- compile
	
	  ./compile.sh
- then go to bx-GooStats folder, this is your working folder. type `make` to compile
	  
	  source GooStats-release/setup.sh
	  cd GooStats-release/GooStats/Modules/naive-Reactor
	  make

- Run your first fit

	  ./fitdata.sh

## Validation guide
- After you finished installation, you can validation it. You should not see any failed test.

	  make test
	  ./autoTest
