CC=cc
CFLAGS+=-Wall -Wextra -pedantic -std=c99 -O3
CFLAGS+=-D_POSIX_C_SOURCE # avoid "fileno" warnings due to c99 standard
CFLAGS+=-DDEBUG # turn debug messages on
CFLAGS+=-DTHP # turn on hugepages
CFLAGS+=-DDOUBLE # turn on double precision

LDFLAGS=-lm -lgomp

all: item_cosine_similarity

item_cosine_similarity: item_cosine_similarity.o hugepages/thp.o
	$(CC) $(CFLAGS) -o $@  $^ $(LDFLAGS)

clean:
	rm -f *.o hugepages/*.o item_cosine_similarity
.PHONY: clean
