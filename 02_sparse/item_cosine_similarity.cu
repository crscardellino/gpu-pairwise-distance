#include <stdbool.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

/* CUDA libraries */
#include <assert.h>
#include <cuda.h>
#include <helper_cuda.h>

#include "hugepages/thp.h"  // huge pages allocation
#include "definitions.h"


/***********************************************
 * Operations for loading matrices and vectors *
 **********************************************/

/* Load the ratings matrix as a sparse CSR matrix */
static void load_ratings_from_mtx(
    const char *fname,
    SparseMatrix dataset)
{
    int i=0, row=0, col=0;
    value_type rating=0.;
    FILE *fstream = fopen(fname, "r");

    if (fstream == NULL) {
        fprintf(stderr, "Error opening the file %s\n", fname);
        exit(EXIT_FAILURE);
    }

    if(fscanf(fstream, "%d %d %d", &dataset->nrows, 
                &dataset->ncols, &dataset->nnz) != 3) {
        fprintf(stderr, "The file is not valid\n");
        exit(EXIT_FAILURE);
    }

    dataset->data = (value_type *) alloc(dataset->nnz, sizeof(value_type));
    dataset->colInd = (int *) alloc(dataset->nnz, sizeof(int));
    dataset->rowPtr = (int *) alloc(dataset->nrows + 1, sizeof(int));
    assert(dataset->data && dataset->colInd && dataset->rowPtr);
    memset(dataset->rowPtr, 0, dataset->nrows + 1 * sizeof(int));

    for (i = 0; i < dataset->nnz; ++i) {
#ifdef DOUBLE
        if(fscanf(fstream, "%d %d %lf", &row, &col, &rating) != 3)
#else
        if(fscanf(fstream, "%d %d %f", &row, &col, &rating) != 3)
#endif
        {
            fprintf(stderr, "The file is not valid\n");
            exit(EXIT_FAILURE);
        }

        dataset->data[i] = rating;
        dataset->colInd[i] = col - 1;
        dataset->rowPtr[row] += 1;
    }
    
    for(row = 1; row < dataset->nrows + 1; ++row)
        dataset->rowPtr[row] += dataset->rowPtr[row-1];

    fclose(fstream);
} 


/* Load the correction vector from the given file */
static void load_correction_vector(
    const char *fname,
    value_type *correction_vector,
    const int vector_size)
{
    int i, read;
    FILE *fstream = fopen(fname, "r");
    
    if(fstream == NULL) {
        fprintf(stderr, "Error opening the file %s\n", fname);
        exit(EXIT_FAILURE);
    }   
    
    for(i = 0; i < vector_size; i++) {
#ifdef DOUBLE
        read = fscanf(fstream, "%le", &correction_vector[i]);
#else
        read = fscanf(fstream, "%e", &correction_vector[i]);
#endif
        if(read == EOF) {
            fprintf(stderr, "Error while reading file %s in element # %d\n", 
                    fname, i);
            fclose(fstream);
            exit(EXIT_FAILURE);
        }
    }

    fclose(fstream);
}


/************************************
 * Cosine similarity operations CPU *
 ***********************************/

/* Returns the vector representing the upper side of the similarity matrix 
 * by measuring cosine similarity pairwise for each row of the item/user matrix */
static void item_cosine_similarity(
    const int *item_user_matrix,
    value_type *similarity_matrix,
    const Dataset dataset)
{
    unsigned int i, u, v, uv, ui, vi;
    value_type num, uden, vden;

    for(u=0; u < dataset->items; u++) {
        for(v=u; v < dataset->items; v++) {
            num=0.;
            uden=0.;
            vden=0.;
            uv = (dataset->items * u) + v - u * (u+1) / 2; 

            for(i = 0; i < dataset->users; i++) {
                ui = u * dataset->users + i;
                vi = v * dataset->users + i;
                num += (value_type) (item_user_matrix[ui] * item_user_matrix[vi]);
                uden += (value_type) (item_user_matrix[ui] * item_user_matrix[ui]);
                vden += (value_type) (item_user_matrix[vi] * item_user_matrix[vi]);
            }
 
            similarity_matrix[uv] = num / (sqrt(uden) * sqrt(vden));
        }
    }
}


/************************************
 * Cosine similarity operations GPU *
 ***********************************/

/* CUDA version. Each thread is in charge of a pair of rows */
__global__ void item_cosine_similarity_cuda(
    const int *item_user_matrix,
    value_type *similarity_matrix,
    const int items,
    const int users)
{
    int i, ui, vi, uv;
    int u = blockIdx.y * blockDim.y + threadIdx.y;
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    value_type num=0., uden=0., vden=0.;

    uv = items * u + v - u * (u + 1) / 2;

    if(v < u || u >= items || v >= items) return;

    for(i=0; i<users; i++) {
        ui = u * users + i;
        vi = v * users + i;
        num += (value_type) (item_user_matrix[ui] * item_user_matrix[vi]);
        uden += (value_type) (item_user_matrix[ui] * item_user_matrix[ui]);
        vden += (value_type) (item_user_matrix[vi] * item_user_matrix[vi]);
    }

    similarity_matrix[uv] = num * rsqrt(uden) * rsqrt(vden);
}


/*****************
 * Main function *
 ****************/

int main(int argc, char **argv) {
    bool correct=true;
    unsigned int i, num_iterations, distance_matrix_size;
    double startTime=0., 
           currentTime=0., 
           refTimeMean=0., 
           optTime=0., 
           previousMean=0.,
           cpuTime=0.,
           globalTime=0.,
           thisTime=0.;
    SparseMatrix dataset;
    value_type *correction_vector, *similarity_matrix, *d_similarity_matrix;

    if (argc < 3 || argc > 4) {
        fprintf(stderr, 
            "usage: ./item_cosine_similarity <user_item_rating_mtx>\
            <correction_vector_vec> [<no_of_iterations>]\n"
        );
        exit(EXIT_FAILURE);
    }
    
    /* start measuring time */
    //thisTime = omp_get_wtime();

    /* Useful for removing noise given by the usage of the machine */
    num_iterations = (argc == 4) ? atoi(argv[3]) : 1;

    /* Reserve space for the SparseMatrix basic structure */
    dataset = (SparseMatrix) malloc(sizeof(struct sDataset));
 
    /* Load ratings dataset from the given mtx file */
    debug("Loading ratings matrix from file %s\n", argv[1]);
    load_ratings_from_mtx(argv[1], dataset);
    debug("Successfully loaded %d total ratings of %d users and %d items\n", 
            dataset->nnz, dataset->ncols, dataset->nrows);

    for(int i=0; i<dataset->nnz; i++)
        fprintf(stderr, "%i ", dataset->data[i]);
    fprintf(stderr, "\n");
    for(int i=0; i<dataset->nnz; i++)
        fprintf(stderr, "%i ", dataset->colInd[i]);
    fprintf(stderr, "\n");
    for(int i=0; i<dataset->nrows + 1; i++)
        fprintf(stderr, "%i ", dataset->colInd[i]);
    fprintf(stderr, "\n");

    free(dataset->data);
    free(dataset->colInd);
    free(dataset->rowPtr);
    free(dataset);

//    /* We use a vector (representing the upper side of a triangular matrix) 
//       in order to make the correction */
//    distance_matrix_size = dataset->items * (dataset->items + 1) / 2;
//    correction_vector = (value_type *) alloc(distance_matrix_size, sizeof(value_type));
//    assert(correction_vector);
//    debug("Loding the correction vector from file %s\n", argv[2]);
//    load_correction_vector(argv[2], correction_vector, distance_matrix_size);
// 
//    /* Calculate the similarity matrix row-wise from the item/user matrix. 
//     * The matrix is represented by a vector of the upper triangular side.
//     * This is what I want to optimize */
//    similarity_matrix = (value_type *) alloc(distance_matrix_size, sizeof(value_type));
//    debug("Calculating items cosine similarity matrices of %d elements\n", 
//            dataset->items);
//
//  cpuTime = omp_get_wtime() - thisTime;
//    globalTime = cpuTime;
//
//    debug("Reference computation will run %d iterations\n", num_iterations);
//
//    for(i = 1; i <= num_iterations; i++) {
//        debug("\rReference iteration number # %d (%d left)", i, num_iterations-i);
//        
//        startTime = omp_get_wtime();
// 
//        /*  What I want to optimize */
//        item_cosine_similarity(item_user_matrix, similarity_matrix, dataset);
//        
//        currentTime = omp_get_wtime() - startTime;
//        previousMean = refTimeMean;
//        refTimeMean += 1.0/(double) i * (currentTime-previousMean);
//    }
//    debug("\nReference computation took %s%.5e%s s, plus %s%.5e%s for the setup.\n", 
//            YELLOW_TEXT, refTimeMean, NO_COLOR, YELLOW_TEXT, cpuTime,
//            NO_COLOR);
//
//    /* CUDA Setup */
//
//    thisTime = omp_get_wtime();
//
//    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
//    dim3 dimGrid((dataset->items + dimBlock.x - 1)/dimBlock.x,
//            (dataset->items + dimBlock.y - 1)/dimBlock.y, 1);
//    checkCudaErrors(cudaMalloc(&d_item_user_matrix, 
//                dataset->items * dataset->users * sizeof(int)));
//    checkCudaErrors(cudaMalloc(&d_similarity_matrix, 
//                distance_matrix_size * sizeof(value_type)));
//    assert(d_item_user_matrix && d_similarity_matrix);
//    checkCudaErrors(cudaMemcpy(d_item_user_matrix, item_user_matrix, 
//                dataset->items * dataset->users * sizeof(int), 
//                cudaMemcpyDefault));
//
//    globalTime += omp_get_wtime() - thisTime;
//
//    debug("Running optimized computation\n");
// 
//    /* Optimized computation */
//        
//    startTime = omp_get_wtime();
//
//    /* Run cuda kernel */
//    checkCudaErrors(cudaMemset(d_similarity_matrix, 0.0f, 
//                distance_matrix_size * sizeof(value_type)));
//    item_cosine_similarity_cuda<<< dimGrid, dimBlock >>>(d_item_user_matrix, 
//            d_similarity_matrix, (int) dataset->items, (int) dataset->users);
//    getLastCudaError("item_cosine_similarity_cuda() kernel failed");
//    checkCudaErrors(cudaDeviceSynchronize());
// 
//    optTime = omp_get_wtime() - startTime;
//
//    thisTime = omp_get_wtime();
//    checkCudaErrors(cudaMemcpy(similarity_matrix, d_similarity_matrix, 
//                distance_matrix_size * sizeof(value_type), cudaMemcpyDefault));
//    globalTime = omp_get_wtime() - thisTime;
//    
//    debug("Optimized computation took %s%.5e%s s plus %s%.5e%s "
//            "for the setup.\n", 
//            YELLOW_TEXT, optTime, NO_COLOR, YELLOW_TEXT, globalTime,
//            NO_COLOR);
//    debug("Rough calculations time speedup: %s%.2fx%s\n",
//          BLUE_TEXT, (refTimeMean)/(optTime), NO_COLOR);
//    debug("Rough wall time speedup: %s%.2fx%s\n",
//          BLUE_TEXT, (refTimeMean+cpuTime)/(optTime+globalTime), NO_COLOR);
// 
//    /* Correction using the previously loaded correction vector */
//    debug("Correction using the given vector and an error of %.0e\n", ERROR);
//    for(i = 0; i < distance_matrix_size; i++) {
//        if(fabs(similarity_matrix[i] - correction_vector[i]) >= ERROR) {
//            correct = false;
//#ifdef DEBUG
//            fprintf(stdout, "%d %.5e %.5e %.5e\n", i, similarity_matrix[i], 
//                    correction_vector[i], 
//                    fabs(similarity_matrix[i] - correction_vector[i]));
//            fflush(stdout);
//#endif
//        }
//    }
//    if(correct){
//        debug("Calculations were %s%s%s\n", GREEN_TEXT, "CORRECT", NO_COLOR);
//    } else {
//        debug("Calculations were %s%s%s\n", RED_TEXT, "WRONG", NO_COLOR);
//    }
//
//    free(dataset);
//    free(ratings);
//    free(item_user_matrix);
//    free(similarity_matrix);
//    free(correction_vector);
//  checkCudaErrors(cudaFree(d_item_user_matrix));
//  checkCudaErrors(cudaFree(d_similarity_matrix));
//
    return EXIT_SUCCESS;
}
