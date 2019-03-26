/*
use 
mpicc -o a.out jf773_Jacobi.c -lm 
to compile
need one argument: dimension of matrix
use 
mpirun -mca plm_rsh_no_tree_spawn 1 --hostfile my_hostfile -np 16 a.out 128 
to execute
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_MAT_VALUE 10
#define MAX(a,b) (((a)>(b))?(a):(b))

/* store the information of abs max element row and column index
*/
struct MaxEle {
    int i;
    int j;
};

/* find the abs max element off the diag in the matrix
*/
struct MaxEle find_abs_max(double** mat, const int n) {
    struct MaxEle ret = {1, 0};
    int i, j;
    double max_val = mat[1][0];
    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            if (abs(mat[i][j]) > abs(max_val)) {
                max_val = mat[i][j];
                ret.i = i;
                ret.j = j;
            }
        }
    }
    return ret;
}

/* create real symmetric matrix
*/
double** create_matrix(const int n) {
    int i, j;
    double** mat = (double **) malloc(n * sizeof(double *));
    for (i = 0; i < n; i++) {
        mat[i] = (double*) malloc(n * sizeof(double));
    }
    srand48(1);
    for (i = 0; i < n; i++) {
        for (j = i; j < n; j++) {
            mat[i][j] = drand48() * MAX_MAT_VALUE;
            mat[j][i] = mat[i][j];
        }
    }

    return mat;
}

/* print matrix
*/
void print_matrix(double** mat, const int n) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%7.2f ", mat[i][j]);
        }
        printf("\n");
    }
}

/* compute the square of the norm of the off diag elements
*/
double off2(double** mat, const int n) {
    double ret = 0;
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i != j) {
                ret += mat[i][j] * mat[i][j];
            }
        }
    }
    return ret;
}

/* compute the square of the norm of the diag elements
*/
double diag_norm_2(double** mat, const int n) {
    double ret = 0;
    int i;
    for (i = 0; i < n; i++) {
        ret += mat[i][i] * mat[i][i];
    }

    return ret;
}

/* update one row of the matrix, distributed block of column to different processes and send the updated value to rank 0
*/
void update_row(double* row_i, double* row_j, double* row_i_updated, double* row_j_updated, MPI_Comm comm, const int n, int is_i, double c, double s) {
    int row_rank = -1;
    int row_size = -1;
    int k;
    if (comm != MPI_COMM_NULL) {
        MPI_Comm_rank(comm, &row_rank);
        MPI_Comm_size(comm, &row_size);
        int block_size = n / row_size;
        double* row_i_block = (double*) malloc(block_size * sizeof(double));
        double* row_j_block = (double*) malloc(block_size * sizeof(double));
        double* row_block_updated = (double*) malloc(block_size * sizeof(double));
        MPI_Scatter(row_i, block_size, MPI_DOUBLE, row_i_block, block_size, MPI_DOUBLE, 0, comm);
        MPI_Scatter(row_j, block_size, MPI_DOUBLE, row_j_block, block_size, MPI_DOUBLE, 0, comm);
        for (k = 0; k < block_size; k++) {
            if (is_i == 1) {
                row_block_updated[k] = c * row_i_block[k] + s * row_j_block[k];
            } else {
                row_block_updated[k] = s * row_i_block[k] - c * row_j_block[k];
            }
        }
        if (is_i == 1) {
            MPI_Gather(row_block_updated, block_size, MPI_DOUBLE, row_i_updated, block_size, MPI_DOUBLE, 0, comm);
            if (row_rank == 0) {
                MPI_Send(row_i_updated, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        } else {
            MPI_Gather(row_block_updated, block_size, MPI_DOUBLE, row_j_updated, block_size, MPI_DOUBLE, 0, comm);
            if (row_rank == 0) {
                MPI_Send(row_j_updated, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        }
        free(row_i_block);
        free(row_j_block);
        free(row_block_updated);
    }
}

/* Jacobi rotation for row i and row j, column i and column j
*/
void jacobi_rotate(double** mat, int idx_i, int idx_j, const int n) {
    int rank;
    int np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    double c, s;
    int i, j;
    int k;
    double* row_i = (double*) malloc(n * sizeof(double));
    double* row_j = (double*) malloc(n * sizeof(double));
    if (rank == 0) {
        i = idx_i;
        j = idx_j;
        if (mat[i][i] == mat[j][j]) {
            c = cos(M_PI / 4);
            s = sin(M_PI / 4);
        } else {
            double tau = (mat[i][i] - mat[j][j]) / (2 * mat[i][j]);
            double t = (tau > 0 ? 1 : -1) / (abs(tau) + sqrt(1 + tau * tau));
            c = 1 / sqrt(1 + t * t);
            s = c * t;
        }
        for (k = 0; k < n; k++) {
            row_i[k] = mat[i][k];
            row_j[k] = mat[j][k];
        }
    }

    MPI_Bcast(&i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&j, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&s, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_i, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_j, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // put even and odd ranks to separate array
    int** ranks = (int **) malloc(2 * sizeof(int *));
    int odd_ranks_size = np / 2;
    int even_ranks_size = odd_ranks_size;
    ranks[0] = (int *) malloc(even_ranks_size * sizeof(int));
    ranks[1] = (int *) malloc(odd_ranks_size * sizeof(int));
    int idx_odd = 0;
    int idx_even = 0;
    for (k = 1; k < np; k++) {
        if (k % 2 == 1) {
            ranks[0][idx_odd] = k;
            idx_odd++;
        } else {
            ranks[1][idx_even] = k;
            idx_even++;
        }
    }

    // create row i and row j group and communicator separately
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group row_i_group;
    MPI_Group row_j_group;
    int group_i_size = (np - 1) / 2;
    int group_j_size = group_i_size;
    MPI_Group_incl(world_group, group_i_size, ranks[0], &row_i_group);
    MPI_Group_incl(world_group, group_j_size, ranks[1], &row_j_group);
    MPI_Comm row_i_comm;
    MPI_Comm row_j_comm;
    MPI_Comm_create_group(MPI_COMM_WORLD, row_i_group, 0, &row_i_comm);
    MPI_Comm_create_group(MPI_COMM_WORLD, row_j_group, 0, &row_j_comm);

    // update row i and row j
    double* row_i_updated = (double*) malloc(n * sizeof(double));
    double* row_j_updated = (double*) malloc(n * sizeof(double));
    update_row(row_i, row_j, row_i_updated, row_j_updated, row_i_comm, n, 1, c, s);
    MPI_Group_free(&row_i_group);
    MPI_Comm_free(&row_i_comm);
    update_row(row_i, row_j, row_i_updated, row_j_updated, row_j_comm, n, 0, c, s);
    MPI_Group_free(&row_j_group);
    MPI_Comm_free(&row_j_comm);

    // in rank 0, update the matrix
    if (rank == 0) {
        MPI_Recv(row_i_updated, n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(row_j_updated, n, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        mat[i][j] = (c * c - s * s) * row_i[j] + s * c * (row_j[j] - row_i[i]);
        mat[j][i] = mat[i][j];
        mat[i][i] = c * c * row_i[i] + 2 * s * c * row_i[j] + s * s * row_j[j];
        mat[j][j] = s * s * row_i[i] - 2 * s * c * row_i[j] + c * c * row_j[j];
        for (k = 0; k < n; k++) {
            if (k != i && k != j) {
                mat[i][k] = row_i_updated[k];
                mat[j][k] = row_j_updated[k];
                mat[k][i] = mat[i][k];
                mat[k][j] = mat[j][k];
            }
        }
    }

    // free resources
    free(row_i_updated);
    free(row_j_updated);
    free(ranks[0]);
    free(ranks[1]);
    free(ranks);
    free(row_i);
    free(row_j);
}

/* use Jacobi method to calculate EVD of a symmetric matrix
*/
void jacobi(double** mat, const int n, const int maxitr, const double eps) {
    struct MaxEle max_ele;
    max_ele = find_abs_max(mat, n);
    double off_norm;
    double diag_norm;
    int rank;
    int i;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        off_norm = off2(mat, n);
        diag_norm = diag_norm_2(mat, n);
    }
    MPI_Bcast(&off_norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&diag_norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int itr_count = 0;
    while (itr_count < maxitr) {
        // check stop condition
        if (itr_count >= 8 && itr_count % 2 == 0) {
            if (off_norm <= diag_norm * eps) {
                break;
            }
        }
        itr_count++;
        jacobi_rotate(mat, max_ele.i, max_ele.j, n);
        if (rank == 0) {
            off_norm = off2(mat, n);
            diag_norm = diag_norm_2(mat, n);
        }
        MPI_Bcast(&off_norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&diag_norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        max_ele = find_abs_max(mat, n);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Missing arguments: dimension of matrix.\n");
        return 0;
    }
    int n = atoi(argv[1]);
    
    MPI_Init(&argc, &argv);
    double** mat = create_matrix(n);

    int rank;
    int np;
    int i;
    double total_time = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    
    if (rank == 0) {
        printf("---------- start ---------\n");
        printf("create real symmetric matrix:\n");
        print_matrix(mat, n);
    }

    int maxitr = MAX(16, log(n));
    total_time -= MPI_Wtime();
    jacobi(mat, n, maxitr, 1e-8);
    total_time += MPI_Wtime();
    if (rank == 0) {
        printf("after using jacobi method, the matrix becomes:\n");
        print_matrix(mat, n);
        printf("eigen values: ");
        for (i = 0; i < n; i++) {
            printf("%.2f ", mat[i][i]);
        }
        printf("\n");
        printf("total time consumed: %f\n", total_time);
        printf("The MPI_Wtime precision is %f seconds\n",MPI_Wtick());
        MPI_Abort(MPI_COMM_WORLD, MPI_SUCCESS);
    }
    MPI_Finalize();
    for (i = 0; i < n; i++) {
        free(mat[i]);
    }
    free(mat);
    return 0;
}