#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "definitions.h"


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

    while((line = fgets(buffer, sizeof(buffer), fstream)) != NULL){
        record = strtok(line, ",");
        for(j=0; j<RATINGS_OFFSET; j++) {
            irecord = atoi(record);

            if (j == 0) {
                dataset->users = max(dataset->users, irecord);
            } else if (j == 1) {
                dataset->items = max(dataset->items, irecord);
            }

            ratings[i * RATINGS_OFFSET + j] = (j==2) ? irecord : irecord - 1;
            record = strtok(NULL, ",");
        }
        i++;
    }

    fclose(fstream);
} 


/* Load the ratings matrix to a item/user matrix */
static void load_item_user_matrix(
    int *item_user_matrix, 
    int *ratings,
    Dataset dataset) 
{
    int i, user, item, rating;

    for(i=0; i < dataset->size; i++) {
        user = ratings[i * RATINGS_OFFSET];
        item = ratings[i * RATINGS_OFFSET + 1];
        rating = ratings[i * RATINGS_OFFSET + 2];
        
        item_user_matrix[item * dataset->users + user] = rating;
    }
}


/* Returns the cosine similarity between two rows of a matrix */
static inline float cosine_similarity_v1(
    int u, 
    int v, 
    int offset,
    int *vector_matrix)
{
    int i;
    float num = 0., uden = 0., vden = 0.;

    for(i=0; i<offset; i++) {
        num += (float) (vector_matrix[u * offset + i] * vector_matrix[v * offset + i]);
        uden += (float) (vector_matrix[u * offset + i] * vector_matrix[u * offset + i]);
        vden += (float) (vector_matrix[v * offset + i] * vector_matrix[v * offset + i]);
    }

    return num / (sqrt(uden) * sqrt(vden));
}


static void item_cosine_similarity_v1(
    int *item_user_matrix,
    float *similarity_matrix,
    Dataset dataset)
{
    int u, v;
    float dist = 0.;

    for(u=0; u < dataset->items; u++) {
        for(v=u; v < dataset->items; v++) {
            dist = cosine_similarity_v1(u, v, dataset->users, item_user_matrix);
 
            similarity_matrix[u * dataset->items + v] = dist;
 
            if (u != v) 
                similarity_matrix[v * dataset->items + u] = dist;
        }
    }
}


int main(int argc, char **argv) {
    unsigned int i=0, num_iterations=0;
    double startTime=0., currentTime=0., timeMean=0., timeVar=0., previousMean=0.;
    Dataset dataset;
    int *ratings, *item_user_matrix; 
    float *similarity_matrix;

    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: ./item_cosine_similarity <user_item_rating_csv> <no_of_ratings> [<no_of_iterations>]\n");
        exit(EXIT_FAILURE);
    }

    dataset = (Dataset) malloc (sizeof(struct sDataset));
    dataset->size = atoi(argv[2]);
    dataset->users = 0;
    dataset->items = 0;

    num_iterations = (argc == 4) ? atoi(argv[3]) : 1;

    ratings = (int *) calloc(dataset->size * RATINGS_OFFSET, sizeof(int));
    debug("Loading ratings matrix from file %s\n", argv[1]);
    load_ratings_from_csv(argv[1], ratings, dataset);
    debug("Successfully loaded %d total ratings of %d users and %d items\n", dataset->size, dataset->users, dataset->items);

    item_user_matrix = (int *) calloc(dataset->items * dataset->users, sizeof(int));
    debug("Loading item/user matrix of size %dx%d\n", dataset->items, dataset->users);
    load_item_user_matrix(item_user_matrix, ratings, dataset);

    similarity_matrix = (float *) calloc(dataset->items * dataset->items, sizeof(float));
    debug("Calculating items cosine similarity matrices of %d elements\n", dataset->items);
    for(i = 1; i <= num_iterations; i++) {
        debug("\rIteration number # %d (%d left)", i, num_iterations-i);
        startTime = omp_get_wtime();
        item_cosine_similarity_v1(item_user_matrix, similarity_matrix, dataset);
        currentTime = omp_get_wtime() - startTime;
        previousMean = timeMean;
        timeMean += 1.0/(double) i * (currentTime-previousMean);
        timeVar  += (currentTime-previousMean)*(currentTime-timeMean);
    }
    debug("\nComputation took %s%.8f%s s (σ²≈%.4f)\n", YELLOW_TEXT, timeMean, NO_COLOR, timeVar);

    free(dataset);
    free(ratings);
    free(item_user_matrix);
	free(similarity_matrix);

    return EXIT_SUCCESS;
}
