NVCC     = nvcc
ARCH     = -arch=sm_61
INCLUDES = -I../../common
CFLAGS   = -O3 -lineinfo

SRCS     = 01_naive 02_tiled 03_vectorized 04_blocktiling2d

all: $(SRCS)

%: %.cu
	$(NVCC) $(ARCH) $(INCLUDES) $(CFLAGS) $< -o $@

debug-%: %.cu
	$(NVCC) $(ARCH) $(INCLUDES) -g -G $< -o $*

clean:
	rm -f $(SRCS)
