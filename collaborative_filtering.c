#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OFFSET 3


struct Dataset {
    int users;
    int movies;
    int size;
};


int max(int a, int b) {
    return (a >= b) ? a : b;
}


void load_csv(char *fname, int *ratings, struct Dataset *dataset){
    char buffer[20];
    char *record, *line;
    int i=0, j=0, irecord=0;
    FILE *fstream = fopen(fname, "r");

    if (fstream == NULL) {
        fprintf(stderr, "Error opening the file %s\n", fname);
        exit(EXIT_FAILURE);
    }

    fprintf(stderr, "Loading ratings matrix from file %s\n", fname);
 
    while((line = fgets(buffer, sizeof(buffer), fstream)) != NULL){
        record = strtok(line, ",");
        for(j=0; j<OFFSET; j++) {
            irecord = atoi(record);

            if (j == 0) {
                dataset->users = max(dataset->users, irecord);
            } else if (j == 1) {
                dataset->movies = max(dataset->movies, irecord);
            }

            ratings[i * OFFSET + j] = irecord;
            record = strtok(NULL, ",");
        }
        i++;
    }
} 

int main(int argc, char **argv) {
    int *ratings;
    struct Dataset dataset;

    if (argc != 3) {
        fprintf(stderr, "usage: ./collaborative_filtering <user_item_rating_matrix> <no_of_ratings>\n");
        exit(EXIT_FAILURE);
    }

    dataset.size = atoi(argv[2]);
    dataset.users = 0;
    dataset.movies = 0;

    ratings = (int *) malloc(dataset.size * OFFSET * sizeof(int));

    load_csv(argv[1], ratings, &dataset);

    fprintf(stderr, "Size of the dataset: %d\n", dataset.size);
    fprintf(stderr, "Users of the dataset: %d\n", dataset.users);
    fprintf(stderr, "Movies of the dataset: %d\n", dataset.movies);

    return EXIT_SUCCESS;
}
