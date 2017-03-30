/*
    Test the result stored in "data_output" against a serial implementation.

    -----
    Compiling:
    Include "Lab4_IO.c" to compile. Set the macro "LAB4_EXTEND" defined in the "Lab4_IO.c" file to include the extended functions
    $ gcc parallel.c Lab4_IO.c -o main -lm 
    $ mpicc -g -lm -o main parallel.c Lab4_IO.c

    -----
    Return values:
    0      result is correct
    1      result is wrong
    2      problem size does not match
    253    no "data_output" file
    254    no "data_input" file
*/
#define LAB4_EXTEND

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mpi.h"
#include "Lab4_IO.h"
#include "timer.h"

#define EPSILON 0.00001
#define DAMPING_FACTOR 0.85

#define THRESHOLD 0.0001

int main (int argc, char* argv[]){
    struct node *nodehead;
    int localcount, nodecount;
    int *num_in_links, *num_out_links;
    double *r_local, *r_pre;

    int i, ii, gi, j;

    double damp_const;

    int comm_size, rank;
    int res = 0;

    double local_error, global_error;

    char flag;

    double start, end;

    // Initialize Data
    if (get_node_stat(&nodecount, &num_in_links, &num_out_links)) return 254;
    if (node_init(&nodehead, num_in_links, num_out_links, 0, nodecount)) return 254;
    r_pre = malloc(nodecount * sizeof(double));

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    localcount = nodecount/comm_size;

    r_local = malloc(localcount * sizeof(double));

    damp_const = (1.0 - DAMPING_FACTOR) / nodecount;

    for ( i = 0; i < localcount; ++i)
        r_local[i] = 1.0 / nodecount;

    // CORE CALCULATION
    GET_TIME(start);
    do{
        MPI_Allgather(
                r_local, localcount, MPI_DOUBLE, 
                r_pre, localcount, MPI_DOUBLE, 
                MPI_COMM_WORLD);

        for ( i = 0; i < localcount; ++i){
            r_local[i] = 0;
            gi = localcount * rank + i;
            for ( j = 0; j < nodehead[gi].num_in_links; ++j){
                ii = nodehead[gi].inlinks[j];
                r_local[i] += r_pre[ii] / num_out_links[ii];
            }
            r_local[i] *= DAMPING_FACTOR;
            r_local[i] += damp_const;
        }
        local_error = rel_error(r_local, r_pre + localcount * rank, localcount);
        MPI_Reduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if(rank == 0){
           flag = global_error >= EPSILON;
        }
        MPI_Bcast(&flag, 1, MPI_BYTE, 0, MPI_COMM_WORLD);
    }while(flag);
    MPI_Gather(r_local, localcount, MPI_DOUBLE, 
            r_pre, localcount, MPI_DOUBLE, 
            0, MPI_COMM_WORLD);
    GET_TIME(end);

    node_destroy(nodehead, nodecount);
    free(num_in_links); free(num_out_links);
    free(r_local);

    if(rank == 0){
        res = Lab4_saveoutput(r_pre, nodecount, end - start);
    } else {
        res = 0;
    }

    free(r_pre); 

    MPI_Finalize();
    printf("R%d: Completed Successfully\n", rank);
    return res; 
}
