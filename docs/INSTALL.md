# Pre-requsite

`GooStats` need installation of [cmake](https://cmake.org/download/)
and [CERN ROOT](https://root.cern/releases/release-62208/). Contact your system admin is you don't have these two
software. If you need to use GPU, you should also make sure CUDA is installed. We have tested the following
combination:
- Springdale Linux release 7.9 (Verona)
- NVCC V10.2.89
- CERN ROOT v6.22.08-centos7-x86_64-gcc4.8
- g++/gcc 7.3.1
- cmake 3.20.0

To save your time, it is recommended that you use ROOT v6.22.08 rather than other versions.

# Installation guide for GooStats

- download recursively all submodules
```shell
git clone --recurse-submodules https://github.com/GooStats/GooStats.git
```

If you have already cloned the project but forgot to add `--recurse-submodules` options, try this command:
```shell
git submodule update --init --recursive
```
More details can be found on [git-scm](https://git-scm.com/book/en/v2/Git-Tools-Submodules).

- If you plan to use GPU, **now you should get access to GPU**. For example, if you
  use a cluster
  managed
  by SLURM,
```shell
srun -qos gpu-test --gres=gpu:1 -A myPorj -c 10 -N 1 -t 1:00:00 --pty bash
```
or if you use PBS
```shell
qsub -q gpu -I
```
You verify that you indeed have access to GPU by executing `nvidia-smi` and see the type of GPU you have.
```shell
nvidia-smi
```
- Configure & generate Makefile / Ninja etc.
```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
```
- compile
```shell
cmake --build build --config Debug -- -j
```
- validate installation
```shell
ctest -C Debug --output-on-failure --verbose
```
