# Linux / Colab shortcut. On Windows use CMake (see README).
#   make            build build/bin/{sw_cpu,sw_gpu,devinfo}
#   make smoke      build + check every GPU version against the golden smoke scores
#   make cpu        CPU only (no CUDA needed)

NVCC ?= nvcc
CXX  ?= g++

NVCC_VER := $(shell $(NVCC) --version 2>/dev/null | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
HAS_SM120 := $(shell echo "$(NVCC_VER)" | awk '{print ($$1 >= 12.8) ? 1 : 0}')

# 75 = T4 (Colab), 86 = RTX 30xx, 89 = L4 / RTX 40xx, 120 = RTX 50xx (needs CUDA >= 12.8)
GENCODE := -gencode arch=compute_75,code=sm_75 \
           -gencode arch=compute_86,code=sm_86 \
           -gencode arch=compute_89,code=sm_89
ifeq ($(HAS_SM120),1)
GENCODE += -gencode arch=compute_120,code=[sm_120,compute_120]
else
GENCODE += -gencode arch=compute_89,code=compute_89
endif

NVCCFLAGS ?= -O3 -lineinfo -std=c++17 $(GENCODE)
CXXFLAGS  ?= -O3 -std=c++17 -Wall -Wextra

BIN := build/bin
GPU_SRC := src/gpu/main_gpu.cu src/gpu/gpu_common.cu $(wildcard src/gpu/gpu_v*.cu)

.PHONY: all cpu gpu smoke clean
all: cpu gpu $(BIN)/devinfo
cpu: $(BIN)/sw_cpu
gpu: $(BIN)/sw_gpu

$(BIN)/sw_cpu: src/cpu/main_cpu.cpp src/cpu/cpu_sw.cpp src/cpu/cpu_sw.h include/sw_common.h
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -Iinclude -Isrc/cpu src/cpu/main_cpu.cpp src/cpu/cpu_sw.cpp -o $@

$(BIN)/sw_gpu: $(GPU_SRC) src/gpu/sw_gpu.h src/gpu/gpu_common.cuh include/sw_common.h
	@mkdir -p $(BIN)
	$(NVCC) $(NVCCFLAGS) -Iinclude -Isrc/gpu $(GPU_SRC) -o $@

$(BIN)/devinfo: tools/devinfo.cu
	@mkdir -p $(BIN)
	$(NVCC) $(NVCCFLAGS) $< -o $@

smoke: all
	python3 scripts/verify.py --in data/smoke.txt --golden data/smoke.golden.txt

clean:
	rm -rf build
