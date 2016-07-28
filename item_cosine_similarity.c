#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "hugepages/thp.h"  // huge pages allocation
#include "definitions.h"


/***********************************************
 * Operations for loading matrices and vectors *
 **********************************************/

/* Load the ratings from a csv file with the format USERID, ITEMID, RATING */
static void load_ratings_from_csv(
    char *fname,
    int *ratings,
    Dataset dataset)
{
    char buffer[20];
    char *record, *line;
    unsigned int i=0, j=0, irecord=0;
    FILE *fstream = fopen(fname, "r");

    if (fstream == NULL) {
        fprintf(stderr, "Error opening the file %s\n", fname);
        exit(EXIT_FAILURE);
    }

    while((line = fgets(buffer, sizeof(buffer), fstream)) != NULL){
        record = strtok(line, ",");
        for(j=0; j<RATINGS_OFFSET; j++) {
            irecord = (unsigned int) atoi(record);

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


/* Load the correction vector from the given file */
static void load_correction_vector(
    char *fname,
    float *correction_vector,
    int vector_size)
{
    int i, read;
    FILE *fstream = fopen(fname, "r");
    
    if(fstream == NULL) {
        fprintf(stderr, "Error opening the file %s\n", fname);
        exit(EXIT_FAILURE);
    }   
    
    for(i = 0; i < vector_size; i++) {
        read = fscanf(fstream, "%e", &correction_vector[i]);
        if(read == EOF) {
            fprintf(stderr, "Error while reading file %s in element # %d\n", fname, i);
            fclose(fstream);
            exit(EXIT_FAILURE);
        }
    }

    fclose(fstream);
}


/* Load the ratings matrix to a item/user matrix */
static void load_item_user_matrix(
    int *item_user_matrix, 
    int *ratings,
    Dataset dataset) 
{
    unsigned int i; 
    int user, item, rating;

    for(i=0; i < dataset->size; i++) {
        user = ratings[i * RATINGS_OFFSET];
        item = ratings[i * RATINGS_OFFSET + 1];
        rating = ratings[i * RATINGS_OFFSET + 2];
        
        item_user_matrix[item * dataset->users + user] = rating;
    }
}


/************************************
 * Cosine similarity operations CPU *
 ***********************************/

/* Returns the vector representing the upper side of the similarity matrix 
 * by measuring cosine similarity pairwise for each row of the item/user matrix */
static void item_cosine_similarity(
    int *item_user_matrix,
    float *similarity_matrix,
    Dataset dataset)
{
    unsigned int i, u, v, uv;
    float num, uden, vden;

    for(u=0; u < dataset->items; u++) {
        for(v=u; v < dataset->items; v++) {
            num=0.;
            uden=0.;
            vden=0.;
            uv = (dataset->items * u) + v - u * (u+1) / 2; 

            for(i = 0; i < dataset->users; i++) {
                num += (float) (item_user_matrix[u * dataset->users + i] * item_user_matrix[v * dataset->users + i]);
                uden += (float) (item_user_matrix[u * dataset->users + i] * item_user_matrix[u * dataset->users + i]);
                vden += (float) (item_user_matrix[v * dataset->users + i] * item_user_matrix[v * dataset->users + i]);
            }
 
            similarity_matrix[uv] = num / (sqrt(uden) * sqrt(vden));
        }
    }
}


/*****************
 * Main function *
 ****************/

int main(int argc, char **argv) {
    bool correct=true;
    unsigned int i, num_iterations, distance_matrix_size;
    float startTime=0., 
           currentTime=0., 
           timeMean=0., 
           timeVar=0., 
           previousMean=0.;
    Dataset dataset;
    int *ratings, *item_user_matrix; 
    float *similarity_matrix, *correction_vector;

    if (argc < 4 || argc > 5) {
        fprintf(stderr, 
            "usage: ./item_cosine_similarity <user_item_rating_csv> <correction_vector_vec> <no_of_ratings> [<no_of_iterations>]\n"
        );
        exit(EXIT_FAILURE);
    }

    /* Initialize the dataset structure. Useful for holding the size of the dataset */
    dataset = (Dataset) malloc (sizeof(struct sDataset));
    dataset->size = (unsigned int) atoi(argv[3]);
    dataset->users = 0;
    dataset->items = 0;

    /* Load ratings dataset from the given csv file */
    ratings = (int *) alloc(dataset->size * RATINGS_OFFSET, sizeof(int));
    debug("Loading ratings matrix from file %s\n", argv[1]);
    load_ratings_from_csv(argv[1], ratings, dataset);
    debug("Successfully loaded %d total ratings of %d users and %d items\n", dataset->size, dataset->users, dataset->items);

    /* We use a vector (representing the upper side of a triangular matrix) in order to make the correction */
    distance_matrix_size = dataset->items * (dataset->items + 1) / 2;
    correction_vector = (float *) alloc(distance_matrix_size, sizeof(float));
    debug("Loding the correction vector from file %s\n", argv[2]);
    load_correction_vector(argv[2], correction_vector, distance_matrix_size);
 
    /* Create the item/user matrix from the previously loaded ratings dataset */
    item_user_matrix = (int *) alloc(dataset->items * dataset->users, sizeof(int));
    debug("Loading item/user matrix of size %dx%d\n", dataset->items, dataset->users);
    load_item_user_matrix(item_user_matrix, ratings, dataset);

    /* Calculate the similarity matrix row-wise from the item/user matrix. 
     * The matrix is represented by a vector of the upper triangular side.
     * This is what I want to optimize */
    similarity_matrix = (float *) alloc(distance_matrix_size, sizeof(float));
    debug("Calculating items cosine similarity matrices of %d elements\n", dataset->items);

    /* Useful for removing noise given by the usage of the machine */
    num_iterations = (argc == 5) ? atoi(argv[4]) : 1;
 
    for(i = 1; i <= num_iterations; i++) {
        debug("\rIteration number # %d (%d left)", i, num_iterations-i);
        
        startTime = omp_get_wtime();
 
        /*  What I want to optimize */
        item_cosine_similarity(item_user_matrix, similarity_matrix, dataset);
        
        currentTime = omp_get_wtime() - startTime;
        previousMean = timeMean;
        timeMean += 1.0/(float) i * (currentTime-previousMean);
        timeVar  += (currentTime-previousMean)*(currentTime-timeMean);
    }
    debug("\nComputation took %s%.8f%s s (σ²≈%.4f)\n", YELLOW_TEXT, timeMean, NO_COLOR, timeVar);

    /* Correction using the previously loaded correction vector */
    debug("Correction using the given vector and an error of %.0e\n", ERROR);
    for(i = 0; i < distance_matrix_size && correct; i++) {
        correct = fabs(similarity_matrix[i] - correction_vector[i]) < ERROR ? true : false;
    }
    debug("Calculations were %s\n", correct ? "CORRECT" : "WRONG !!!");

    free(dataset);
    free(ratings);
    free(item_user_matrix);
    free(similarity_matrix);
    free(correction_vector);

    return EXIT_SUCCESS;
}
