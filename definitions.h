#ifndef __DEFINITIONS_H
#define __DEFINITIONS_H

#define RATINGS_OFFSET 3

/* Return the max between two values */
#define max(a, b) ((a >= b) ? a : b)

/* Struct to hold information regarding a dataset (size, no of users and no of items) */
typedef struct sDataset {
    unsigned int size;
    unsigned int users;
    unsigned int items;
} * Dataset;

/* Result misscalculation tolerance */
#define ERROR 0.00001

/* Debug messages printing */
#ifdef DEBUG
# define debug(format,...) {                 \
    fprintf(stderr, format, ## __VA_ARGS__); \
    fflush(stderr);                          \
  }
#else
# define debug(format,...)
#endif

/* Color text printing codes */
#define  RED_TEXT       "\33[22;31m"
#define  RED_B_TEXT     "\33[1;31m"
#define  GREEN_TEXT     "\33[22;32m"
#define  GREEN_B_TEXT   "\33[1;32m"
#define  YELLOW_TEXT    "\33[22;33m"
#define  YELLOW_B_TEXT  "\33[1;33m"
#define  BLUE_TEXT      "\33[22;34m"
#define  BLUE_B_TEXT    "\33[1;34m"
#define  NO_COLOR       "\33[0m"

#endif /* __DEFINITIONS_H */
