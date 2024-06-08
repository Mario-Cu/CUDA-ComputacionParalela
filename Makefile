# Location of CUDA
CUDA_DIR ?= /usr/local/cuda
CUDA_LIB_DIR := -L$(CUDA_DIR)/lib64
CUDA_INC_DIR := -I$(CUDA_DIR)/include

# Compilers
GCC	 := g++
NVCC := $(CUDA_DIR)/bin/nvcc -ccbin $(GCC)

# Common Includes and Libs for CUDA
CUDA_INC_COMMON = -I../common
CUDA_LD_LIBS := -lcudart

# Internal Flags
CC_FLAGS     := -O3
LD_FLAGS     := -lm
NVCC_FLAGS   := -m64
GENCODE_FLAGS = -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90

ALL_CC_FLAGS :=
ALL_CC_FLAGS += $(NVCC_FLAGS)

ALL_LD_FLAGS :=
ALL_LD_FLAGS += $(ALL_CC_FLAGS)

ALL_CC_FLAGS += --threads 0 --std=c++11 -w

FILE = 

build: $(FILE)

$(FILE).o: $(FILE).cu
		   $(NVCC) $(CUDA_INC_DIR) $(CUDA_INC_COMMON) $(ALL_CC_FLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(FILE): $(FILE).o
		 $(NVCC) $(CC_FLAGS) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $+

run:
	./$(FILE) 70 0.1 0.3 0.35 100 5 5 30 15 50 15 80 M 123123
clean:
	rm -f *.o $(FILE)
