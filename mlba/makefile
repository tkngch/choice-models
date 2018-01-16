CC=g++
CFLAGS=-O2 -Wall -Wextra -std=c++11 -lgsl -lcblas
LINKPATH=$(HOME)/lib/cpp
FORMAT=clang-format -style="LLVM" -i

.PHONY: default format

default: libmlba.so libmlba.a test

test: test.o mlba.o
	$(CC) $(CFLAGS) $^ -o $@

test.o: test.hpp test.cpp
	$(CC) $(CFLAGS) -c test.cpp -o $@

libmlba.a: mlba.hpp mlba.cpp
	$(CC) $(CFLAGS) -static -fPIC -c mlba.cpp -o mlba_.o
	ar rvs $@ mlba_.o

libmlba.so: mlba.o
	$(CC) $(CFLAGS) -shared $^ -o $@

mlba.o: mlba.hpp mlba.cpp
	$(CC) $(CFLAGS) -fPIC -c mlba.cpp -o $@

link:
	rm -f $(LINKPATH)/libmlba.so $(LINKPATH)/mlba.hpp
	ln -s $(realpath .)/libmlba.so $(LINKPATH)/libmlba.so
	ln -s $(realpath .)/mlba.hpp $(LINKPATH)/mlba.hpp

format: $(wildcard *.hpp) $(wildcard *.cpp)
	$(FORMAT) $^
