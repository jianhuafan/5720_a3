#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    int msg_len = 1 << 8;
    char msg[msg_len];
    MPI_Init(NULL, NULL);
    int rank;
    int world;
    double start_time, end_time;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    int i;
    int k;

    if (rank == 0) {
        memset(msg, 'a', sizeof(msg) - 1);
        start_time = MPI_Wtime();
        MPI_Send(msg, strlen(msg) + 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(msg, msg_len, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        end_time = MPI_Wtime();
        printf("%s\n", msg);
        printf("%f seconds\n", end_time - start_time);
        printf("The MPI_Wtime precision is %f seconds\n",MPI_Wtick());
    }
    if (rank == 1) {
        MPI_Recv(msg, msg_len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(msg, strlen(msg) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}