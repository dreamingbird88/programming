# A simple makefile example, from
# http://mrbook.org/blog/tutorials/make/
# run:
#   make -f makefile
#   make all
#   make clean
C=g++
CFLAGS=-c -Wall -std=gnu99
LDFLAGS=
SOURCES=input.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=input

all: $(SOURCES) $(EXECUTABLE)
	    
$(EXECUTABLE): $(OBJECTS) 
	    $(CC) $(LDFLAGS) $(OBJECTS) -o $@

.c.o:
	    $(CC) $(CFLAGS) $< -o $@
clean:
	rm -f $(OBJECTS) ${EXECUTABLE}
