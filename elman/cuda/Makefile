AR ?= ar
CXX ?= g++
NVCC ?= nvcc -ccbin $(CXX)
PYTHON ?= python

ifeq ($(OS),Windows_NT)
LIBHASTY := hasty.lib
CUDA_HOME ?= $(CUDA_PATH)
AR := lib
AR_FLAGS := /nologo /out:$(LIBHASTY)
NVCC_FLAGS := -x cu -Xcompiler "/MD"
else
LIBHASTY := libhasty.a
CUDA_HOME ?= /usr/local/cuda
AR ?= ar
AR_FLAGS := -crv $(LIBHASTY)
NVCC_FLAGS := -std=c++17 -x cu -Xcompiler -fPIC
endif

# CUTLASS path (for B2B GEMM fusion)
CUTLASS_HOME ?= /home/erikg/cutlass

LOCAL_CFLAGS := -I/usr/include/eigen3 -I$(CUDA_HOME)/include -Ilib -O3
LOCAL_LDFLAGS := -L$(CUDA_HOME)/lib64 -L. -lcudart -lcublas -lcufft
GPU_ARCH_FLAGS := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89

# CUTLASS include paths for B2B GEMM
CUTLASS_CFLAGS := -I$(CUTLASS_HOME)/include -I$(CUTLASS_HOME)/tools/util/include -Ilib/cutlass_b2b

# E-Series CUDA kernels (archived kernels in lib/archive/)
CUDA_OBJS := \
	lib/stock_elman_gpu.o \
	lib/mamba_gated_elman_gpu.o \
	lib/softsign_elman_gpu.o \
	lib/diagonal_state_elman_gpu.o \
	lib/slot_elman_gpu.o \
	lib/lowrank_slot_elman_gpu.o \
	lib/lowrank_elman_gpu.o \
	lib/pure_lowrank_elman_gpu.o \
	lib/pure_lowrank_elman_fused_gpu.o \
	lib/diagonal_elman_gpu.o \
	lib/b2b_lowrank_elman_gpu.o \
	lib/circulant_elman_gpu.o \
	lib/monarch_elman_gpu.o \
	lib/scaled_lowrank_elman_gpu.o \
	lib/hybrid_elman_gpu.o \
	lib/multiscale_elman_gpu.o \
	lib/selective_elman_gpu.o \
	lib/selective_gated_elman_gpu.o \
	lib/selective_wh_elman_gpu.o \
	lib/matrix_state_elman_gpu.o \
	lib/haware_gate_elman_gpu.o \
	lib/simplified_gate_elman_gpu.o \
	lib/mamba2_informed_elman_gpu.o \
	lib/structured_elman_gpu.o \
	lib/structured_elman_attention_gpu.o \
	lib/dual_memory_elman_gpu.o \
	lib/e23c_chunked_gpu.o \
	lib/e23c_v2_gpu.o \
	lib/e24_single_gemm_gpu.o \
	lib/e25_entmax_gpu.o \
	lib/e26_parallel_gpu.o

.PHONY: all hasty hasty_tf hasty_pytorch libhasty_tf examples benchmarks clean

all: hasty hasty_tf hasty_pytorch examples benchmarks

# Pattern rule for compiling CUDA files - enables parallel builds with make -j
lib/%.o: lib/%.cu.cc
	$(NVCC) $(GPU_ARCH_FLAGS) -c $< -o $@ $(NVCC_FLAGS) $(LOCAL_CFLAGS)

# Special rule for B2B GEMM kernel - needs CUTLASS headers
lib/b2b_lowrank_elman_gpu.o: lib/b2b_lowrank_elman_gpu.cu.cc
	$(NVCC) $(GPU_ARCH_FLAGS) -c $< -o $@ $(NVCC_FLAGS) $(LOCAL_CFLAGS) $(CUTLASS_CFLAGS)

# Special rule for Circulant kernel - needs cuFFT
lib/circulant_elman_gpu.o: lib/circulant_elman_gpu.cu.cc
	$(NVCC) $(GPU_ARCH_FLAGS) -c $< -o $@ $(NVCC_FLAGS) $(LOCAL_CFLAGS)

# hasty target now depends on object files and just creates archive
hasty: $(CUDA_OBJS)
	$(AR) $(AR_FLAGS) $(CUDA_OBJS)

libhasty_tf: hasty
	$(eval TF_CFLAGS := $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'))
	$(eval TF_LDFLAGS := $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'))
	$(CXX) -std=c++11 -c frameworks/tf/lstm.cc -o frameworks/tf/lstm.o $(LOCAL_CFLAGS) $(TF_CFLAGS) -fPIC
	$(CXX) -std=c++11 -c frameworks/tf/gru.cc -o frameworks/tf/gru.o $(LOCAL_CFLAGS) $(TF_CFLAGS) -fPIC
	$(CXX) -std=c++11 -c frameworks/tf/layer_norm.cc -o frameworks/tf/layer_norm.o $(LOCAL_CFLAGS) $(TF_CFLAGS) -fPIC
	$(CXX) -std=c++11 -c frameworks/tf/layer_norm_gru.cc -o frameworks/tf/layer_norm_gru.o $(LOCAL_CFLAGS) $(TF_CFLAGS) -fPIC
	$(CXX) -std=c++11 -c frameworks/tf/layer_norm_indrnn.cc -o frameworks/tf/layer_norm_indrnn.o $(LOCAL_CFLAGS) $(TF_CFLAGS) -fPIC
	$(CXX) -std=c++11 -c frameworks/tf/layer_norm_lstm.cc -o frameworks/tf/layer_norm_lstm.o $(LOCAL_CFLAGS) $(TF_CFLAGS) -fPIC
	$(CXX) -std=c++11 -c frameworks/tf/indrnn.cc -o frameworks/tf/indrnn.o $(LOCAL_CFLAGS) $(TF_CFLAGS) -fPIC
	$(CXX) -std=c++11 -c frameworks/tf/support.cc -o frameworks/tf/support.o $(LOCAL_CFLAGS) $(TF_CFLAGS) -fPIC
	$(CXX) -shared frameworks/tf/*.o libhasty.a -o frameworks/tf/libhasty_tf.so $(LOCAL_LDFLAGS) $(TF_LDFLAGS) -fPIC

# Dependencies handled by setup.py
hasty_tf:
	@$(eval TMP := $(shell mktemp -d))
	@cp -r . $(TMP)
	@cat build/common.py build/setup.tf.py > $(TMP)/setup.py
	@(cd $(TMP); $(PYTHON) setup.py -q bdist_wheel)
	@cp $(TMP)/dist/*.whl .
	@rm -rf $(TMP)

# Dependencies handled by setup.py
hasty_pytorch:
	@$(eval TMP := $(shell mktemp -d))
	@cp -r . $(TMP)
	@cat build/common.py build/setup.pytorch.py > $(TMP)/setup.py
	@(cd $(TMP); $(PYTHON) setup.py -q bdist_wheel)
	@cp $(TMP)/dist/*.whl .
	@rm -rf $(TMP)

dist:
	@$(eval TMP := $(shell mktemp -d))
	@cp -r . $(TMP)
	@cp build/MANIFEST.in $(TMP)
	@cat build/common.py build/setup.tf.py > $(TMP)/setup.py
	@(cd $(TMP); $(PYTHON) setup.py -q sdist)
	@cp $(TMP)/dist/*.tar.gz .
	@rm -rf $(TMP)
	@$(eval TMP := $(shell mktemp -d))
	@cp -r . $(TMP)
	@cp build/MANIFEST.in $(TMP)
	@cat build/common.py build/setup.pytorch.py > $(TMP)/setup.py
	@(cd $(TMP); $(PYTHON) setup.py -q sdist)
	@cp $(TMP)/dist/*.tar.gz .
	@rm -rf $(TMP)

examples: hasty
	$(CXX) -std=c++11 examples/lstm.cc $(LIBHASTY) $(LOCAL_CFLAGS) $(LOCAL_LDFLAGS) -o hasty_lstm -Wno-ignored-attributes
	$(CXX) -std=c++11 examples/gru.cc $(LIBHASTY) $(LOCAL_CFLAGS) $(LOCAL_LDFLAGS) -o hasty_gru -Wno-ignored-attributes

benchmarks: hasty
	$(CXX) -std=c++11 benchmarks/benchmark_lstm.cc $(LIBHASTY) $(LOCAL_CFLAGS) $(LOCAL_LDFLAGS) -o benchmark_lstm -Wno-ignored-attributes -lcudnn
	$(CXX) -std=c++11 benchmarks/benchmark_gru.cc $(LIBHASTY) $(LOCAL_CFLAGS) $(LOCAL_LDFLAGS) -o benchmark_gru -Wno-ignored-attributes -lcudnn

clean:
	rm -fr benchmark_lstm benchmark_gru hasty_lstm hasty_gru hasty_*.whl hasty_*.tar.gz
	find . \( -iname '*.o' -o -iname '*.so' -o -iname '*.a' -o -iname '*.lib' \) -delete
