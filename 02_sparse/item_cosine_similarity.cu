#include <stdbool.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <omp.h>

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
static inline int load_ratings_from_mtx(
    const char *fname,
    SparseMatrix dataset)
{
    int i=0, row=0, col=0, nnz=0;
    value_type rating=0.;
    FILE *fstream = fopen(fname, "r");

    if (fstream == NULL) {
        fprintf(stderr, "Error opening the file %s\n", fname);
        exit(EXIT_FAILURE);
    }

    if(fscanf(fstream, "%d %d %d", &dataset->nrows, 
                &dataset->ncols, &nnz) != 3) {
        fprintf(stderr, "The file is not valid\n");
        exit(EXIT_FAILURE);
    }

    dataset->data = (value_type *) alloc(nnz, sizeof(value_type));
    dataset->colInd = (int *) alloc(nnz, sizeof(int));
    dataset->rowPtr = (int *) alloc((dataset->nrows + 1), sizeof(int));
    assert(dataset->data && dataset->colInd && dataset->rowPtr);
    memset(dataset->rowPtr, 0, (dataset->nrows + 1) * sizeof(int));

    for (i = 0; i < nnz; ++i) {
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

    return nnz;
} 


/* Load the correction vector from the given file */
static inline void load_correction_vector(
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
static inline void item_cosine_similarity(
    const SparseMatrix dataset,
    value_type *similarity_matrix)
{
#pragma omp parallel for default(shared)
    for(int u=0; u < dataset->nrows; u++) {
#pragma omp parallel for default(shared)
        for(int v=u; v < dataset->nrows; v++) {
            int i = dataset->rowPtr[u];
            int j = dataset->rowPtr[v];
            int uv = (dataset->nrows * u) + v - u * (u+1) / 2; 
            value_type num=0., uden=0., vden=0.;
 
            while(i < dataset->rowPtr[u+1] && j < dataset->rowPtr[v+1]) {
                if(dataset->colInd[i] == dataset->colInd[j])
                    num += (value_type) (dataset->data[i] * dataset->data[j]);

                if(dataset->colInd[i] <= dataset->colInd[j]) i++;

                if(dataset->colInd[j] < dataset->colInd[i]) j++;
            }

            for(i = dataset->rowPtr[u]; i < dataset->rowPtr[u+1]; i++) {
                uden += (value_type) (dataset->data[i] * dataset->data[i]);
            }

            for(i = dataset->rowPtr[v]; i < dataset->rowPtr[v+1]; i++) {
                vden += (value_type) (dataset->data[i] * dataset->data[i]);
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
    const value_type *data,
    const int *colInd,
    const int *rowPtr,
    const int nrows,
    value_type *similarity_matrix)
{
    int i, j, uv;
    int u = blockIdx.y * blockDim.y + threadIdx.y;
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    value_type num=0., uden=0., vden=0.;

    uv = nrows * u + v - u * (u + 1) / 2;

    if(v < u || u >= nrows || v >= nrows) return;

    i = rowPtr[u];
    j = rowPtr[v];

    while(i < rowPtr[u+1] && j < rowPtr[v+1]) {
        if(colInd[i] == colInd[j])
            num += (value_type) (data[i] * data[j]);

        if(colInd[i] <= colInd[j]) i++;

        if(colInd[j] < colInd[i]) j++;
    }

    for(i = rowPtr[u]; i < rowPtr[u+1]; i++) {
        uden += (value_type) (data[i] * data[i]);
    }

    for(i = rowPtr[v]; i < rowPtr[v+1]; i++) {
        vden += (value_type) (data[i] * data[i]);
    }

    similarity_matrix[uv] = num * rsqrt(uden) * rsqrt(vden);
}


/*****************
 * Main function *
 ****************/

int main(int argc, char **argv) {
    bool correct=true;
    int i, num_iterations, distance_matrix_size, nnz;
    double startTime=0., 
           currentTime=0., 
           refTimeMean=0., 
           optTime=0., 
           previousMean=0.,
           cpuTime=0.,
           globalTime=0.,
           thisTime=0.;
    SparseMatrix dataset;
    int *d_colInd, *d_rowPtr;
    value_type *correction_vector, *similarity_matrix, *d_data, *d_similarity_matrix;

    if (argc < 3 || argc > 4) {
        fprintf(stderr, 
            "usage: ./item_cosine_similarity <user_item_rating_mtx>\
            <correction_vector_vec> [<no_of_iterations>]\n"
        );
        exit(EXIT_FAILURE);
    }
    
    /* start measuring time */
    thisTime = omp_get_wtime();

    /* Useful for removing noise given by the usage of the machine */
    num_iterations = (argc == 4) ? atoi(argv[3]) : 1;

    /* Reserve space for the SparseMatrix basic structure */
    dataset = (SparseMatrix) malloc(sizeof(struct sSparseMatrix));
 
    /* Load ratings dataset from the given mtx file */
    debug("Loading ratings matrix from file %s\n", argv[1]);
    nnz = load_ratings_from_mtx(argv[1], dataset);
    debug("Successfully loaded %d total ratings of %d users and %d items\n", 
            nnz, dataset->ncols, dataset->nrows);

    /* We use a vector (representing the upper side of a triangular matrix) 
       in order to make the correction */
    distance_matrix_size = dataset->nrows * (dataset->nrows + 1) / 2;
    correction_vector = (value_type *) alloc(distance_matrix_size, sizeof(value_type));
    assert(correction_vector);
    debug("Loding the correction vector from file %s\n", argv[2]);
    load_correction_vector(argv[2], correction_vector, distance_matrix_size);
 
    /* Calculate the similarity matrix row-wise from the item/user matrix. 
     * The matrix is represented by a vector of the upper triangular side.
     * This is what I want to optimize */
    similarity_matrix = (value_type *) alloc(distance_matrix_size, sizeof(value_type));
    debug("Calculating items cosine similarity matrices of %d elements\n", 
            dataset->nrows);

    cpuTime = omp_get_wtime() - thisTime;
    globalTime = cpuTime;

    debug("Reference computation will run %d iterations\n", num_iterations);

    for(i = 1; i <= num_iterations; i++) {
        debug("\rReference iteration number # %d (%d left)", i, num_iterations-i);
        
        startTime = omp_get_wtime();
 
        /*  What I want to optimize */
        item_cosine_similarity(dataset, similarity_matrix);
        
        currentTime = omp_get_wtime() - startTime;
        previousMean = refTimeMean;
        refTimeMean += 1.0/(double) i * (currentTime-previousMean);
    }
    debug("\nReference computation took %s%.5e%s s, plus %s%.5e%s for the setup.\n", 
            YELLOW_TEXT, refTimeMean, NO_COLOR, YELLOW_TEXT, cpuTime,
            NO_COLOR);

    /* CUDA Setup */

    thisTime = omp_get_wtime();

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dimGrid((dataset->nrows + dimBlock.x - 1)/dimBlock.x,
            (dataset->nrows + dimBlock.y - 1)/dimBlock.y, 1);

    checkCudaErrors(cudaMalloc(&d_data, nnz * sizeof(value_type)));
    checkCudaErrors(cudaMalloc(&d_colInd, nnz * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_rowPtr, (dataset->nrows + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_similarity_matrix, 
                distance_matrix_size * sizeof(value_type)));
    assert(d_data && d_colInd && d_rowPtr && d_similarity_matrix);
 
    checkCudaErrors(cudaMemcpy(d_data, dataset->data, nnz * sizeof(value_type), 
                cudaMemcpyDefault));
    checkCudaErrors(cudaMemcpy(d_colInd, dataset->colInd, nnz * sizeof(int),
                cudaMemcpyDefault));
    checkCudaErrors(cudaMemcpy(d_rowPtr, dataset->rowPtr, 
                (dataset->nrows + 1) * sizeof(int), cudaMemcpyDefault));
    checkCudaErrors(cudaMemset(d_similarity_matrix, 0.0f, 
                distance_matrix_size * sizeof(value_type)));

    globalTime += omp_get_wtime() - thisTime;

    debug("Running optimized computation\n");
 
    /* Optimized computation */
        
    startTime = omp_get_wtime();

    /* Run cuda kernel */
    item_cosine_similarity_cuda<<< dimGrid, dimBlock >>>(d_data,
            d_colInd, d_rowPtr, dataset->nrows, d_similarity_matrix);
    getLastCudaError("item_cosine_similarity_cuda() kernel failed");
    checkCudaErrors(cudaDeviceSynchronize());
 
    optTime = omp_get_wtime() - startTime;

    thisTime = omp_get_wtime();
    checkCudaErrors(cudaMemcpy(similarity_matrix, d_similarity_matrix, 
                distance_matrix_size * sizeof(value_type), cudaMemcpyDefault));
    globalTime += omp_get_wtime() - thisTime;
    
    debug("Optimized computation took %s%.5e%s s plus %s%.5e%s "
            "for the setup.\n", 
            YELLOW_TEXT, optTime, NO_COLOR, YELLOW_TEXT, globalTime,
            NO_COLOR);
    debug("Rough calculations time speedup: %s%.2fx%s\n",
          BLUE_TEXT, (refTimeMean)/(optTime), NO_COLOR);
    debug("Rough wall time speedup: %s%.2fx%s\n",
          BLUE_TEXT, (refTimeMean+cpuTime)/(optTime+globalTime), NO_COLOR);
 
    /* Correction using the previously loaded correction vector */
    debug("Correction using the given vector and an error of %.0e\n", ERROR);
    for(i = 0; i < distance_matrix_size; i++) {
        if(fabs(similarity_matrix[i] - correction_vector[i]) >= ERROR) {
            correct = false;
#ifdef DEBUG
            fprintf(stdout, "%d %.5e %.5e %.5e\n", i, similarity_matrix[i], 
                    correction_vector[i], 
                    fabs(similarity_matrix[i] - correction_vector[i]));
            fflush(stdout);
#endif
        }
    }
    if(correct){
        debug("Calculations were %s%s%s\n", GREEN_TEXT, "CORRECT", NO_COLOR);
    } else {
        debug("Calculations were %s%s%s\n", RED_TEXT, "WRONG", NO_COLOR);
    }

    free(dataset->data);
    free(dataset->colInd);
    free(dataset->rowPtr);
    free(dataset);
    free(similarity_matrix);
    free(correction_vector);
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_colInd));
    checkCudaErrors(cudaFree(d_rowPtr));
    checkCudaErrors(cudaFree(d_similarity_matrix));

    return EXIT_SUCCESS;
}
