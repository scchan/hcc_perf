
CC=gcc
CXX=g++

COMMON_FLAGS+=-I./include
COMMON_FLAGS+=-O3
COMMON_FLAGS+=-g
COMMON_FLAGS+=-Wall
#COMMON_FLAGS+=-DDEBUG


CFLAGS+=$(COMMON_FLAGS)
CXXFLAGS+=$(COMMON_FLAGS)


all: libStackTimer.so test_stack_timer


libStackTimer.so: StackTimer.o
	$(CXX) $^ -shared -o $@

StackTimer.o: StackTimer.cpp
	$(CXX) -fPIC $(CXXFLAGS) -c $^


test_stack_timer: test_stack_timer.c libStackTimer.so
	$(CC) $@.c -lStackTimer $(CFLAGS) -L. -o $@

install: all
	cp libStackTimer.so test_stack_timer ./bin

clean:
	rm -f *.o libStackTimer.so test_stack_timer
