#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#define NO_OF_CORES 2

int main()
{
  double *b, d;
  int i, it, m, n;
  double r, t, *x, *xnew;

  m = 200; // no of itereations
  n = 500000; // size of the matrix

  // Serial operation
  clock_t start_time_s, end_time_s;   // for serial
  start_time_s = clock();

  b = (double *)malloc(n * sizeof(double));
  x = (double *)malloc(n * sizeof(double));
  xnew = (double *)malloc(n * sizeof(double));


  {

    for (i = 0; i < n; i++)
    {
      b[i] = 0.0;
    }

    b[n - 1] = (double)(n + 1);

    for (i = 0; i < n; i++)
    {
      x[i] = 0.0;
    }
  }

  for (it = 0; it < m; it++)
  {

    {

      for (i = 0; i < n; i++)
      {
        xnew[i] = b[i];
        if (0 < i)
        {
          xnew[i] = xnew[i] + x[i - 1];
        }
        if (i < n - 1)
        {
          xnew[i] = xnew[i] + x[i + 1];
        }
        xnew[i] = xnew[i] / 2.0;
      }

      d = 0.0;

      for (i = 0; i < n; i++)
      {
        d = d + pow(x[i] - xnew[i], 2);
      }

      for (i = 0; i < n; i++)
      {
        x[i] = xnew[i];
      }
      r = 0.0;

      for (i = 0; i < n; i++)
      {
        t = b[i] - 2.0 * x[i];
        if (0 < i)
        {
          t = t + x[i - 1];
        }
        if (i < n - 1)
        {
          t = t + x[i + 1];
        }
        r = r + t * t;
      }

      {
        if (it < 10 || m - 10 < it)
        {
          // printf("  %8d  %14.6g  %14.6g\n", it, sqrt(d), sqrt(r));
        }
        if (it == 9)
        {
          // printf("  Omitting intermediate results.\n");
        }
      }
    }
  }
  // printf("\n Part of final solution estimate:\n");
  // for (i = 0; i < 10; i++)
  // {
  //   printf("  %8d  %14.6g\n", i, x[i]);
  // }
  // printf("...\n");
  // for (i = n - 11; i < n; i++)
  // {
  //   printf("  %8d  %14.6g\n", i, x[i]);
  // }

  // end of serial execution
  end_time_s = clock();
  double execution_time_s = (double)(end_time_s - start_time_s) / CLOCKS_PER_SEC;

  // free the memory
  free(b);
  free(x);
  free(xnew);


  // Parallel execution using the same parameters
  // new mem allocation is needed as the memory has been freed at the end of the serial execution
  b = (double *)malloc(n * sizeof(double));
  x = (double *)malloc(n * sizeof(double));
  xnew = (double *)malloc(n * sizeof(double));

  // start of the parallel execution
  double start_time_p, end_time_p; // for parallel operation
  start_time_p = omp_get_wtime();

  // set the number of threads
  omp_set_num_threads(NO_OF_CORES);


#pragma omp parallel private(i)
  {
#pragma omp for
    for (i = 0; i < n; i++)
    {
      b[i] = 0.0;
    }

    b[n - 1] = (double)(n + 1);

#pragma omp for
    for (i = 0; i < n; i++)
    {
      x[i] = 0.0;
    }
  }

  for (it = 0; it < m; it++)
  {
#pragma omp parallel private(i, t)
    {

#pragma omp for
      for (i = 0; i < n; i++)
      {
        xnew[i] = b[i];
        if (0 < i)
        {
          xnew[i] = xnew[i] + x[i - 1];
        }
        if (i < n - 1)
        {
          xnew[i] = xnew[i] + x[i + 1];
        }
        xnew[i] = xnew[i] / 2.0;
      }

      d = 0.0;
#pragma omp for reduction(+ : d)
      for (i = 0; i < n; i++)
      {
        d = d + pow(x[i] - xnew[i], 2);
      }

#pragma omp for
      for (i = 0; i < n; i++)
      {
        x[i] = xnew[i];
      }

      r = 0.0;
#pragma omp for reduction(+ : r)
      for (i = 0; i < n; i++)
      {
        t = b[i] - 2.0 * x[i];
        if (0 < i)
        {
          t = t + x[i - 1];
        }
        if (i < n - 1)
        {
          t = t + x[i + 1];
        }
        r = r + t * t;
      }

#pragma omp master
      {
        if (it < 10 || m - 10 < it)
        {
          // printf("  %8d  %14.6g  %14.6g\n", it, sqrt(d), sqrt(r));
        }
        if (it == 9)
        {
          // printf("Omitting intermediate results.\n");
        }
      }
    }
  }

  // printf("\nPart of final solution estimate:\n");
  // for (i = 0; i < 10; i++)
  // {
  //   printf("  %8d  %14.6g\n", i, x[i]);
  // }
  // printf("...\n");
  // for (i = n - 11; i < n; i++)
  // {
  //   printf("  %8d  %14.6g\n", i, x[i]);
  // }

  // end time for parallel execution
  end_time_p = omp_get_wtime();
  double execution_time_p = end_time_p - start_time_p;

  free(b);
  free(x);
  free(xnew);

  // Execution time for serial and parallel operations
  printf("\nSerial Execution Time (seconds): %f\n", execution_time_s);
  printf("Execution time for Parallel (seconds): %f\n\n", execution_time_p); 

  return 0;
}
