// Estudiante: Josue Peña Atencio - 8935601

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char *argv[]) {
  int rank, size;
  long long i = 0, n = 0;
  double w, mypi, pi, start, end;
  double PI25DT = 3.141592653589793238462643;
   
  MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  
    if (rank == 0) {
      printf("Enter the number of segments for integration: ");
      scanf("%llu", &n);
    }
    
    start = MPI_Wtime(); 
    MPI_Barrier(MPI_COMM_WORLD);
    
      /* Tell all processes, the number of segments you want */
      MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
      
      mypi = 0.0;
      w = 1.0 / (double) n;
      for (i = rank+1; i <= n; i += size)
        mypi += w*sqrt(1 - (((double) i / n) * ((double) i / n)));

      MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    
    if (rank == 0) {
      printf("pi is approximately %.20f, Error is %.20f%%\n", 4*pi, 100*fabs((4 * pi) - PI25DT));
      printf("The calculation took %.3f seconds.\n", end-start);
    }

      
  MPI_Finalize();
  return 0;
}
