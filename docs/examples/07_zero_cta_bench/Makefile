NCCL_HOME ?= /usr
CUDA_HOME ?= /usr/local/cuda

CXX ?= g++
CXXFLAGS = -O3 -std=c++17 -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include
LDFLAGS = -L$(NCCL_HOME)/lib/x86_64-linux-gnu -lnccl -L$(CUDA_HOME)/lib64 -lcudart -lpthread

TARGET = zero_cta_bench

all: $(TARGET)

$(TARGET): main.cc
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

test: $(TARGET)
	CUDA_VISIBLE_DEVICES=6,7 LD_LIBRARY_PATH=/usr/local/cuda/compat:$(NCCL_HOME)/lib/x86_64-linux-gnu ./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all test clean
