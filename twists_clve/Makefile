
#nvcc -arch=all --default-stream per-thread -Xcudafe --diag_suppress=20208,--diag_suppress=set_but_not_used -Wno-deprecated-gpu-targets -o twists_cuda twists_cuda.cu -O2 -lflint -lm -L/home/egalois/JB_cuda -ltwists_dirichlet_character -I/usr/local/include/flint -I/home/egalois/JB_cuda -I/home/egalois/NVIDIA_CUDA-11.4_Samples/common/inc/

SRCDIR=source

CC=gcc
CLIBS=-lflint -lm
CINCLUDE=include

CNAME=twists_dirichlet_character
CDIR=$(SRCDIR)/$(CNAME)

CSRCS=$(CDIR)/$(CNAME).c
CHDRS=$(CDIR)/$(CINCLUDE)/$(CNAME).h
COBJS=$(CNAME).o
CTARGET=lib$(CNAME).a

AR=ar
ARFLAGS=rcv
RANLIB=ranlib

NVCC=nvcc
NVCCARCH=-gencode arch=compute_61,code=[sm_61,compute_61]
NVCCSTRM=--default-stream per-thread
NVCCWARN=-Xcudafe --diag_suppress=20208
NVCCOTHR=-Wno-deprecated-gpu-targets
NVCCFLAGS=
NVCCFLAGS+=$(NVCCARCH)
NVCCFLAGS+= $(NVCCSTRM)
NVCCFLAGS+= $(NVCCWARN)
NVCCFLAGS+= $(NVCCOTHR)

NVCCLIBS=$(CLIBS)
NVCCINCLUDE=include

NVCCNAME=twists_clve

NVCCSRCS=$(SRCDIR)/$(NVCCNAME).cu
NVCCHDRS=$(SRCDIR)/$(NVCCINCLUDE)/$(NVCCNAME).cuh
NVCCOBJS=$(CNAME)
NVCCTARGET=$(NVCCNAME)

all: $(NVCCTARGET)
	$(MAKE) clean

$(NVCCTARGET):$(NVCCSRCS) $(NVCCHDRS) $(CTARGET)
	$(NVCC) $(NVCCFLAGS) -o $(NVCCTARGET) $(NVCCSRCS) -O2 $(NVCCLIBS) -L. -l$(NVCCOBJS) -I$(SRCDIR)/$(NVCCINCLUDE) -I$(CDIR)/$(CINCLUDE)

$(CTARGET):$(COBJS)
	$(AR) $(ARFLAGS) $@ $(COBJS)
	$(RANLIB) $@

$(COBJS):$(CSRCS) $(CHDRS)
	$(CC) -c -O2 -I$(CDIR)/$(CINCLUDE) $(CSRCS) $(CLIBS)

clean:
	rm -f $(COBJS)
	rm -f $(CTARGET)

new:
	$(MAKE) clean
	$(MAKE) all
