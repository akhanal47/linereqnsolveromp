#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#define NO_OF_CORES 1

// make the matrix symmetrical definate 
void symm_def(double *A, double *C, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double sum = 0;
            for (int k = 0; k < n; k++)
            {
                sum += A[i * n + k] * A[j * n + k];
            }
            C[i * n + j] = sum;
        }
    }
}

// serial method for cholesky decomposition
double *cholesky(double *A, int n)
{
    double *L = (double *)calloc(n * n, sizeof(double));
    if (L == NULL)
        exit(EXIT_FAILURE);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < (i + 1); j++)
        {
            double s = 0;
            for (int k = 0; k < j; k++)
            {
                s += L[i * n + k] * L[j * n + k];
            }
            L[i * n + j] = (i == j) ? sqrt(A[i * n + i] - s) : (1.0 / L[j * n + j] * (A[i * n + j] - s));
        }
    }
    return L;
}

double inner_sum(double *li, double *lj, int n)
{
    double s = 0;
    for (int i = 0; i < n; i++)
    {
        s += li[i] * lj[i];
    }
    return s;
}


// double *choleskyp1(double *A, int n)
// {
//     double *L = (double *)calloc(n * n, sizeof(double));
//     if (L == NULL)
//         exit(EXIT_FAILURE);
//     for (int j = 0; j < n; j++)
//     {
//         double s = inner_sum(&L[j * n], &L[j * n], j);
//         L[j * n + j] = sqrt(A[j * n + j] - s);

//         #pragma omp parallel for schedule(dynamic, 8)
//         for (int i = j + 1; i < n; i++)
//         {
//             double s = inner_sum(&L[j * n], &L[i * n], j);
//             L[i * n + j] = (1.0 / L[j * n + j] * (A[i * n + j] - s));
//         }
//     }
//     return L;
// }

double *choleskyp2(double *A, int n)
{
    double *L = (double *)calloc(n * n, sizeof(double));
    if (L == NULL)
        exit(EXIT_FAILURE);

    #pragma omp parallel for schedule(dynamic, 1) // Adjust the schedule type and chunk size
    for (int j = 0; j < n; j++)
    {
        double s = 0.0;

        // Compute the diagonal element
        for (int k = 0; k < j; k++)
        {
            s += L[j * n + k] * L[j * n + k];
        }
        L[j * n + j] = sqrt(A[j * n + j] - s);

        // Compute the lower triangular elements
        #pragma omp parallel for schedule(dynamic, 1) // Adjust the schedule type and chunk size
        for (int i = j + 1; i < n; i++)
        {
            double s = 0.0;
            for (int k = 0; k < j; k++)
            {
                s += L[i * n + k] * L[j * n + k];
            }
            L[i * n + j] = (1.0 / L[j * n + j]) * (A[i * n + j] - s);
        }
    }

    return L;
}

void show_matrix(double *A, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            printf("%2.5f ", A[i * n + j]);
        printf("\n");
    }
}


// Function to solve the linear system Ly = b and L^Tx = y
void solveLinearSystem(double *L, double *b, double *x, int n)
{
    // Forward substitution: Solve Ly = b for y
    for (int i = 0; i < n; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < i; j++)
        {
            sum += L[i * n + j] * x[j];
        }
        x[i] = (b[i] - sum) / L[i * n + i];
    }

    // Backward substitution: Solve L^Tx = y for x
    for (int i = n - 1; i >= 0; i--)
    {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++)
        {
            sum += L[j * n + i] * x[j];
        }
        x[i] = (x[i] - sum) / L[i * n + i];
    }
}


int main()
{
    int n = 1000;
    double start_time_p1, end_time_p1; // for parallel p1
    double start_time_p2, end_time_p2; // for parallel p2
    clock_t start_time_s, end_time_s;   // for serial
    double *m3 = (double *)malloc(sizeof(double) * n * n);

    // generate the random matrix
    srand(time(NULL)); // Initialize the random number generator
    for (int i = 0; i < n; i++)
    {
        for (int j = i; j < n; j++)
        {
            double element = 1.0 * rand() / RAND_MAX;
            m3[i * n + j] = element;
            m3[j * n + i] = element;
        }
    }
    double *m4 = (double *)malloc(sizeof(double) * n * n);
    // make a positive-definite matrix
    symm_def(m3, m4, n); 

    // adding stability to the matrix
    double *m8 = (double *)malloc(sizeof(double) * n * n);
    for (int i = 0; i < n; i++)
    {
        for (int j = i; j < n; j++)
        {
            double element = 1.0 * rand() / RAND_MAX;
            m8[i * n + j] = element;
            m8[j * n + i] = element;
        }
    }
    // serial operation
    // as only the decomposition is timed, we will compare the execution time of the serial and parallel decomposition procss
    start_time_s = clock();  // Start time for the program
    double *sd = cholesky(m4, n);
    end_time_s = clock();
    double execution_time_s = (double)(end_time_s - start_time_s) / CLOCKS_PER_SEC;

    // // parallel operation with multi threads
    // start_time_p1 = omp_get_wtime();
    // double *pd1 = choleskyp1(m4, n);
    // end_time_p1 = omp_get_wtime();
    // double execution_time_p1 = end_time_p1 - start_time_p1;

    // set no of threads
    omp_set_num_threads(NO_OF_CORES);

    // parallel operation with multi threads
    start_time_p2 = omp_get_wtime();
    double *pd2 = choleskyp2(m4, n);
    end_time_p2 = omp_get_wtime();
    double execution_time_p2 = end_time_p2 - start_time_p2;

    // // display the decomposed matrix
    // show_matrix(sd,n);

    // solve Ly = b and L^Tx = y
    double *b = (double *)malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++)
    {
        b[i] = 1.0 * rand() / RAND_MAX;
    }

    double *y = (double *)malloc(sizeof(double) * n);
    double *x = (double *)malloc(sizeof(double) * n);

    // solve the intermediate linear systems
    solveLinearSystem(sd, b, y, n);

    // // Display the solution vector y
    // printf("Solution vector y:\n");
    // for (int i = 0; i < n; i++)
    // {
    //     printf("%2.5f\n", y[i]);
    // }

    solveLinearSystem(sd, y, x, n);

    // // Display the solution vector x, this is the final soulation for the form Ax=B
    // printf("Solution vector x:\n");
    // for (int i = 0; i < n; i++)
    // {
    //     printf("%2.5f\n", x[i]);
    // }

    // print the time taken for solving the cholesky decomposition task
    printf("\nSerial Execution time (seconds): %f\n", execution_time_s);
    // printf("\nExecution time for Parallel Form 1 (seconds): %f\n", execution_time_p1); 
    // printf("Execution time for Parallel Form (seconds): %f\n\n", execution_time_p2); 
    printf("Execution time for Parallel Form (seconds): %f\n\n", execution_time_p2); 


    free(m3);
    free(m4);
    free(m8);
    free(sd);
    free(b);
    free(y);
    free(x);
    return 0;
}