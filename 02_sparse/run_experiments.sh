#!/usr/bin/env bash
set -e

# echo "COMPILING CODE DENSE MATRIX"
# echo "---------------------------"
# cd 01_cuda
# make clean
# CUCFLAGS=-DGPUONLY make
# echo
# 
# echo "RUNNING CODE"
# echo "------------"
# echo

# for dfile in ml100k ml1M ml10M ml20M
# do
#     echo "RUNNING EXPERIMENTS FOR $dfile"
#     echo "------------------------------"
#     echo
# 
#     matrix=../data/$dfile.mtx
#     vector=../data/$dfile.vec
# 
#     OMP_NUM_THREADS=12 ./item_cosine_similarity $matrix $vector 5 2>&1 | tee ../report/data/cuda/cuda_12_threads_$dfile.txt 
#     echo
# done


echo "RUNNING EXPERIMENTS"
echo "-------------------"

for dfile in ml100k ml1M ml10M ml20M
do
    echo "RUNNING EXPERIMENTS FOR $dfile"
    echo "------------------------------"
    echo

    matrix=../data/$dfile.mtx
    vector=../data/$dfile.vec

    for i in 2 4 8 16 32
    do 
        echo "COMPILING AND RUNNING BLOCK OF SIZE $i"
        echo "--------------------------------------"
        make clean
        CUCFLAGS="-DGPUONLY -DBLOCK_SIZE=$i" make 
        ./item_cosine_similarity $matrix $vector 2>&1 | tee ../report/data/sparse/sparse_${i}_block_$dfile.txt 
        echo
    done
done
