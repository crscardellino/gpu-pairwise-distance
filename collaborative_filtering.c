#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MOVIELENS_ROWS 1000209
#define MOVIELENS_COLS 3
#define MOVIELENS_USERS 6040
#define MOVIELENS_MOVIES 3952


void load_csv(char *fname, int ratings[MOVIELENS_ROWS][MOVIELENS_COLS]){
    char buffer[20];
    char *record, *line;
    int i=0, j=0;
    FILE *fstream = fopen(fname, "r");

    if (fstream == NULL) {
        fprintf(stderr, "Error opening the file %s\n", fname);
        exit(EXIT_FAILURE);
    }

    fprintf(stderr, "Loading ratings matrix from file %s\n", fname);
    while((line = fgets(buffer, sizeof(buffer), fstream)) != NULL){
        record = strtok(line, ",");
        while(record != NULL) {
            ratings[i][j++] = atoi(record);
            record = strtok(NULL, ",");
        }
        i++;
    }
} 

int main(int argc, char **argv) {
    int ratings[MOVIELENS_ROWS][MOVIELENS_COLS];

    if (argc != 2) {
        fprintf(stderr, "usage: ./collaborative_filtering <user_item_rating_matrix>\n");
        exit(EXIT_FAILURE);
    }

    load_csv(argv[1], ratings);

    for(int i=0; i < MOVIELENS_ROWS; i++) {
        for(int j=0; j < MOVIELENS_COLS; j++) {
            fprintf(stdout, "%d\t", ratings[i][j]);
        }
        fprintf(stdout, "\n");
    }

    return EXIT_SUCCESS;
}
