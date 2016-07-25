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


void load_ratings_from_csv(char *fname, int *ratings, struct Dataset *dataset){
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

            ratings[i * OFFSET + j] = (j==2) ? irecord : irecord - 1;
            record = strtok(NULL, ",");
        }
        i++;
    }

    fclose(fstream);
} 

void load_user_movie_matrix(int *user_movie_matrix, int *ratings, struct Dataset dataset) {
    int i, user, movie, rating;

    fprintf(stderr, "Loading user movie matrix\n");
    for(i=0; i < dataset.size; i++) {
        user = ratings[i * OFFSET];
        movie = ratings[i * OFFSET + 1];
        rating = ratings[i * OFFSET + 2];
        
        user_movie_matrix[user * dataset.movies + movie] = rating;
    }
}

void item_similarity(int *items_matrix, double *similarity_matrix) {

}

int main(int argc, char **argv) {
    int *ratings, *user_movie_matrix;
    struct Dataset dataset;

    if (argc != 3) {
        fprintf(stderr, "usage: ./collaborative_filtering <user_item_rating_matrix> <no_of_ratings>\n");
        exit(EXIT_FAILURE);
    }

    dataset.size = atoi(argv[2]);
    dataset.users = 0;
    dataset.movies = 0;

    ratings = (int *) malloc(dataset.size * OFFSET * sizeof(int));
    load_ratings_from_csv(argv[1], ratings, &dataset);

    user_movie_matrix = (int *) malloc(dataset.users * dataset.movies * sizeof(int));
    load_user_movie_matrix(user_movie_matrix, ratings, dataset);

    free(ratings);
    free(user_movie_matrix);

    return EXIT_SUCCESS;
}
