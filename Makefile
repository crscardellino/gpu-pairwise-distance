CC=cc
CFLAGS=-Wall -Wextra -pedantic -std=c99 -g
CFLAGS+=-D_POSIX_C_SOURCE  # avoid "fileno" warnings due to c99 standard
CFLAGS+=-DDEBUG   # turn debug messages on
CFLAGS+=-DNDEBUG  # turn assertions off

LDFLAGS=-lm

all: user_cosine_similarity

user_cosine_similarity: user_cosine_similarity.o
	$(CC) $(CFLAGS) -o $@  $^ $(LDFLAGS)

clean:
	rm -f *.o user_cosine_similarity
.PHONY: clean
