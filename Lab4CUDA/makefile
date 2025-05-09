CC = nvcc
CFLAGS = -O3 -g

all: main

main: main.cu lenet.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

lenet.o: lenet.cu lenet.h
	$(CC) $(CFLAGS) -c lenet.cu -lm

clean:
	rm -f main *.o
