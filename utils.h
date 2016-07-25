#ifndef __UTILS_H
#define __UTILS_H

#define OFFSET 3

/* Return the max between two values */
#define max(a, b) ((a >= b) ? a : b)

/* Struct to hold information regarding a dataset (size, no of users and no of items) */
typedef struct sDataset {
    int size;
    int users;
    int items;
} * Dataset;

#endif /* __UTILS_H */
