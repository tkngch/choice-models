CC=g++
CFLAGS=-O2 -Wall -Wextra -std=c++11
MVTNORM=./libMvtnorm
MVTFILES=$(wildcard $(MVTNORM)/*.o $(MVTNORM)/*.h)
LINKPATH=/home/takao/lib/cpp
FORMAT=clang-format -style="LLVM" -i


.PHONY: default link format

default: test_mdft

test_mdft: test_mdft.o libmdft.a
	$(CC) $(CFLAGS) $^ -o $@

test_mdft.o: test_mdft.hpp test_mdft.cpp
	cp $(MVTFILES) ./
	$(CC) $(CFLAGS) -c test_mdft.cpp -o $@
	rm $(notdir $(MVTFILES))

libmdft.a: mdft.o
	cp $(MVTFILES) ./
	ar rvs $@ $^ mvtnorm.o mvt.o randomF77.o
	rm $(notdir $(MVTFILES))

mdft.o: mdft.hpp mdft.cpp
	cp $(MVTFILES) ./
	$(CC) $(CFLAGS) -fPIC -static -c mdft.cpp -o $@
	rm $(notdir $(MVTFILES))

link:
	rm -f $(LINKPATH)/libmdft.a
	ln -s $(realpath .)/libmdft.a $(LINKPATH)/libmdft.a
	rm -f $(LINKPATH)/mvtnorm.h
	ln -s $(realpath .)/$(MVTNORM)/mvtnorm.h $(LINKPATH)/mvtnorm.h

format: $(wildcard *.hpp) $(wildcard *.cpp)
	$(FORMAT) $^
