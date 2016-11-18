CC=g++
CFLAGS=-O2 -Wall -Wextra -std=c++11
FORMAT=clang-format -style="LLVM" -i

.PHONY: default format

default: libaam.so test

test.o: test.hpp test.cpp
	$(CC) $(CFLAGS) -c test.cpp -o $@

test: test.o aam.o
	$(CC) $(CFLAGS) $^ -o $@

aam.o: aam.hpp aam.cpp
	$(CC) $(CFLAGS) -fPIC -c aam.cpp -o $@

libaam.so: aam.o
	$(CC) $(CFLAGS) -shared $^ -o $@

format: $(wildcard *.hpp) $(wildcard *.cpp)
	$(FORMAT) $^
