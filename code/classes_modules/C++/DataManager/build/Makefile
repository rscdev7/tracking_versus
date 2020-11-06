CC				= g++
CFLAGS			= -g -Wall -std=c++0x -O3 -march=native
INCLUDE 		= -I.

TARGETS 		= ../test/tester.cpp ../lib/libDataManager.hpp ../lib/libDataManager.cpp
EXEC    		= data_manager
EXEC_CANC    	= ../bin/data_manager

all:
	$(CC) $(INCLUDE) $(CFLAGS) $(TARGETS) -o $(EXEC)
	mv    $(EXEC)    ../bin

exec:
	../bin/$(EXEC)

.PHONY: clean
clean:
	rm    -f $(EXEC_CANC)
