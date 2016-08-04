CC=cc
CFLAGS+=-Wall -Wextra -pedantic -std=c99 -O3
CFLAGS+=-D_POSIX_C_SOURCE # avoid "fileno" warnings due to c99 standard
CFLAGS+=-DDEBUG # turn debug messages on
CFLAGS+=-DTHP # turn on hugepages

# Device compiler
CUC=/opt/cuda/7.5/bin/nvcc
CUCFLAGS=-O -I/opt/cuda/7.5/samples/common/inc -arch=sm_52 --prec-div=true --prec-sqrt=true
CUCFLAGS+=-D_POSIX_C_SOURCE # avoid "fileno" warnings due to c99 standard
CUCFLAGS+=-DDEBUG # turn debug messages on
CUCFLAGS+=-DTHP # turn on hugepages


CXX=g++-4.9 
CXXFLAGS=-Wall -Wextra -march=native -O3
CXXFLAGS+=-Wno-variadic-macros  # These are standard from C++11 onwards
CXXFLAGS+=-Wno-long-long  # These are standard from C++11 onwards
CXXFLAGS+=-Wno-unused-function

LDFLAGS=-lm -lgomp

all: item_cosine_similarity_cuda

item_cosine_similarity: item_cosine_similarity.o hugepages/thp.o
	$(CC) $(CFLAGS) -o $@  $^ $(LDFLAGS)

item_cosine_similarity_cuda: item_cosine_similarity_cuda.o hugepages/thp.o
	$(CUC) $(CUCFLAGS) --compiler-options="$(CXXFLAGS)" -o $@ $^ $(LDFLAGS)

%.o: %.cu
	$(CUC) $(CUCFLAGS) --compiler-options="$(CXXFLAGS)" -o $@ -c $<

clean:
	rm -f *.o item_cosine_similarity item_cosine_similarity_cuda
.PHONY: clean
