CC=cc
CFLAGS=-Wall -Wextra -pedantic -std=c99
CFLAGS+=-D_POSIX_C_SOURCE  # avoid "fileno" warnings due to c99 standard
CFLAGS+=-DDEBUG   # turn debug messages on
CFLAGS+=-DNDEBUG  # turn assertions off

LDFLAGS=-lm

all: collaborative_filtering

collaborative_filtering: collaborative_filtering.o
	$(CC) $(CFLAGS) -o $@  $^ $(LDFLAGS)

clean:
	rm -f *.o collaborative_filtering
.PHONY: clean
