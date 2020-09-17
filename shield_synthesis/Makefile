ifndef CUDD_PATH
#CUDD_PATH = /usr/local
parent_dir = $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
CUDD_PATH = $(parent_dir)/cudd-3.0.0
endif

EXECUTABLE = shield_synthesizer
SOURCES    = $(wildcard *.cpp)
OBJECTS    = $(patsubst %,%,${SOURCES:.cpp=.o})
CXX        = g++
FLAGS      = -Wall -c -std=c++11 -O3 -march=native
#LIBS       = -L$(CUDD_PATH)/lib -l cudd
#INCLUDE    = -I$(CUDD_PATH)/include
LIBS       = -L$(CUDD_PATH)/cudd/.libs -l cudd
INCLUDE    = -I$(CUDD_PATH)/cudd

all: $(EXECUTABLE) 

%.o: %.cpp
	$(CXX) $(FLAGS) -o $@ $< $(INCLUDE)

#link Objects
$(EXECUTABLE) : $(OBJECTS)
	$(CXX) -o $@ $^ $(LIBS)
	chmod u+x $(EXECUTABLE)

clean:
	rm -f *.o
	rm $(EXECUTABLE)
