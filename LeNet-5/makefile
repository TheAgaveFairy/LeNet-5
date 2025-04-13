CC = gcc
CFLAGS = -O3 -g

all: main

main: main.c lenet.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

lenet.o: lenet.c lenet.h
	$(CC) $(CFLAGS) -c lenet.c -lm

clean:
	rm -f main *.o
