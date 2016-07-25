#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils.h"


/* Load the ratings from a csv file with the format USERID, ITEMID, RATING */
static void load_ratings_from_csv(
    char *fname,
    int *ratings,
    Dataset dataset)
{
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
                dataset->items = max(dataset->items, irecord);
            }

            ratings[i * OFFSET + j] = (j==2) ? irecord : irecord - 1;
            record = strtok(NULL, ",");
        }
        i++;
    }

    fprintf(stderr, "Successfully loaded %d total ratings of %d users and %d items\n", dataset->size, dataset->users, dataset->items);

    fclose(fstream);
} 


/* Load the ratings matrix to a user/item matrix */
static void load_user_item_matrix(
    int *user_item_matrix, 
    int *ratings,
    Dataset dataset) 
{
    int i, user, item, rating;

    fprintf(stderr, "Loading user item matrix\n");
    for(i=0; i < dataset->size; i++) {
        user = ratings[i * OFFSET];
        item = ratings[i * OFFSET + 1];
        rating = ratings[i * OFFSET + 2];
        
        user_item_matrix[user * dataset->items + item] = rating;
    }
}


/* Returns the cosine similarity between two rows of a matrix */
static inline float cosine_similarity_v1(
    int u, 
    int v, 
    int items,
    int *vector_matrix)
{
    int i;
    float num = 0., uden = 0., vden = 0.;

    for(i=0; i<items; i++) {
        num += (float) (vector_matrix[u * items + i] * vector_matrix[v * items + i]);
        uden += (float) (vector_matrix[u * items + i] * vector_matrix[u * items + i]);
        vden += (float) (vector_matrix[v * items + i] * vector_matrix[v * items + i]);
    }

    return num / (sqrt(uden) * sqrt(vden));
}


static void user_cosine_similarity_v1(
    int *user_item_matrix,
    float *similarity_matrix,
    Dataset dataset)
{
    int u, v;
    float dist = 0.;

    fprintf(stderr, "Calculating users cosine similarity matrix\n");
    for(u=0; u < dataset->users; u++) {
        for(v=u; v < dataset->users; v++) {
            dist = cosine_similarity_v1(u, v, dataset->items, user_item_matrix);
 
            similarity_matrix[u * dataset->users + v] = dist;
 
            if (u != v) 
                similarity_matrix[v * dataset->users + u] = dist;
        }
    }
}


int main(int argc, char **argv) {
    Dataset dataset;
    int *ratings, *user_item_matrix; 
    float *similarity_matrix;

    if (argc != 3) {
        fprintf(stderr, "usage: ./user_cosine_similarity <user_item_rating_csv> <no_of_ratings>\n");
        exit(EXIT_FAILURE);
    }

    dataset = (Dataset) malloc (sizeof(struct sDataset));
    dataset->size = atoi(argv[2]);
    dataset->users = 0;
    dataset->items = 0;

    ratings = (int *) calloc(dataset->size * OFFSET, sizeof(int));
    load_ratings_from_csv(argv[1], ratings, dataset);

    user_item_matrix = (int *) calloc(dataset->users * dataset->items, sizeof(int));
    load_user_item_matrix(user_item_matrix, ratings, dataset);

    similarity_matrix = (float *) calloc(dataset->users * dataset->users, sizeof(float));
    user_cosine_similarity_v1(user_item_matrix, similarity_matrix, dataset);

    free(dataset);
    free(ratings);
    free(user_item_matrix);

    return EXIT_SUCCESS;
}
