# Pre-requsite

`GooStats` need installation of `cmake` and `CERN ROOT`. Contact your system admin is you don't have these two software.

# Installation guide for GooStats

- Create a folder for `GooStats` and go to that folder

	  mkdir GooStats-release
	  cd GooStats-release
- download the installation script and run it. **Make sure you have internet at this step.** It will download the `GooStats` and `googletest`, create a bunch of folders, and symbol link the `compile.sh`

	  git clone https://github.com/GooStats/GooStats.git
	  . GooStats/setup/download.sh
- run the `compile.sh`. **Make sure you have GPU at this step**. This script will compile `GooFit` and `googletest`, and also create a script for setting up environment `setup.sh`

	  . compile.sh
  - Tips: to get GPU, if you are working on a cluster, launch an interactive job to a cluster node equipped with GPU: `qsub -q gpu -I` or `srun -p myPartition --gres=gpu:1 -A myPorj  -c 40 -N 1 -t 1:00:00 --pty bash`

- compile example project `naive-Reactor`
	  
	  source setup.sh
	  cd GooStats/Modules/naive-Reactor
	  make

- Run your first fit

	  ./fitdata.sh

## Validation guide
- After you finished installation, you can validation it. You should not see any failed test.

	  make test
	  ./autoTest

you should see

    [==========] Running 4 tests from 1 test case.
    [----------] Global test environment set-up.
    [----------] 4 tests from BestFitFixture
    [ RUN      ] BestFitFixture.TAUP_npmt_exact
    [       OK ] BestFitFixture.TAUP_npmt_exact (10 ms)
    [ RUN      ] BestFitFixture.TAUP_npmt_near
    [       OK ] BestFitFixture.TAUP_npmt_near (8 ms)
    [ RUN      ] BestFitFixture.Asimov_exact
    [       OK ] BestFitFixture.Asimov_exact (15 ms)
    [ RUN      ] BestFitFixture.toyMC_exact
    [       OK ] BestFitFixture.toyMC_exact (152 ms)
    [----------] 4 tests from BestFitFixture (185 ms total)
     
    [----------] Global test environment tear-down
    [==========] 4 tests from 1 test case ran. (185 ms total)
    [  PASSED  ] 4 tests.
