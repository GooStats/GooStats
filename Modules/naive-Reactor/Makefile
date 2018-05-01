BUILD :=  ../../../build_GooStats
BUILD_NLL :=  ../../../build_GooStats_NLL
BUILD_RPF :=  ../../../build_GooStats_RPF
config := ${BUILD}/CMakeCache.txt
CMKFLS := $(shell find ../.. -name CMakeLists.txt)
all: config
	bash -c "cd ${BUILD}; make -j reactor;"
	ln -sf ${BUILD}/Modules/naive-Reactor/reactor
nll: 
	mkdir -p ${BUILD_NLL}
	bash -c "cd ${BUILD_NLL}; cmake ../GooStats -DNLL_CHECK=1"
	bash -c "cd ${BUILD_NLL}; make -j;"
	ln -sf ${BUILD_NLL}/Modules/naive-Reactor/reactor

rpf: 
	mkdir -p ${BUILD_RPF}
	bash -c "cd ${BUILD_RPF}; cmake ../GooStats -DRPF_CHECK=1"
	bash -c "cd ${BUILD_RPF}; make -j;"
	ln -sf ${BUILD_RPF}/Modules/naive-Reactor/reactor

clean: config
	bash -c "cd ${BUILD}/Modules/naive-Reactor; make clean"

config: ${CMKFLS}
	bash -c 'cd ${BUILD}; cmake ../GooStats'
