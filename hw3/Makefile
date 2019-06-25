CC=gcc
CFLAGS_GCC=-Wall -std=gnu11
CFLAGS=-g -lineinfo -O3
CUDAFLAGS=-maxrregcount=32
LDFLAGS=-libverbs

all: client server

client: client.o common.o
	$(CC) $^ $(LDFLAGS) -o $@

server: server.o common.o
	nvcc $^ $(CUDAFLAGS) $(LDFLAGS) -o $@

%.o: %.cu
	nvcc $(CFLAGS) $(CUDAFLAGS) -c -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) $(CFLAGS_GCC) -c -o $@ $<

clean:
	rm -f *.o client server
