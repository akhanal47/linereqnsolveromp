#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define TOLERANCE 1e-20
#define NO_OF_CORES 8

void initializeMatrix(float **A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= n; j++) {
            A[i][j] = 1.0 / (i + j + 1);
        }
    }
}

void gaussianElimination_s(float **A, int n) {
    for (int j = 0; j < n; j++) {
        // Partial Pivoting
        int maxRow = j;
        for (int i = j + 1; i < n; i++) {
            if (fabs(A[i][j]) > fabs(A[maxRow][j])) {
                maxRow = i;
            }
        }

        // Swap rows
        for (int k = 0; k <= n; k++) {
            float temp = A[j][k];
            A[j][k] = A[maxRow][k];
            A[maxRow][k] = temp;
        }

        for (int i = 0; i < n; i++) {
            if (i > j) {
                if (fabs(A[j][j]) < TOLERANCE) {
                    // Handle the case where the pivot element is effectively zero
                    printf("Pivot element is effectively zero. Unable to proceed.\n");
                    exit(EXIT_FAILURE);
                }

                float c = A[i][j] / A[j][j];
                for (int k = 0; k <= n; k++) {
                    A[i][k] = A[i][k] - c * A[j][k];
                }
            }
        }
    }
}

void gaussianElimination_p(float **A, int n) {
    #pragma omp parallel
    {
        int i, j, k;
        #pragma omp for
        for (j = 0; j < n; j++) {
            // Partial Pivoting
            int maxRow = j;
            {
                for (i = j + 1; i < n; i++) {
                    if (fabs(A[i][j]) > fabs(A[maxRow][j])) {
                        maxRow = i;
                    }
                }

                // Swap rows
                for (k = 0; k <= n; k++) {
                    float temp = A[j][k];
                    A[j][k] = A[maxRow][k];
                    A[maxRow][k] = temp;
                }
            }

            for (i = 0; i < n; i++) {
                if (i > j) {
                    if (fabs(A[j][j]) < TOLERANCE) {
                        // Handle the case where the pivot element is effectively zero
                        {
                            printf("Pivot element is effectively zero. Unable to proceed.\n");
                            exit(EXIT_FAILURE);
                        }
                    }

                    float c = A[i][j] / A[j][j];
                    for (k = 0; k <= n; k++) {
                        A[i][k] = A[i][k] - c * A[j][k];
                    }
                }
            }
        }
    }
}

void printSolution(float *x, int n) {
    printf("\nSolution:\n");
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %f\n", i, x[i]);
    }
}

void freeMemory(float **A, float *x, int n) {
    free(x);
    for (int i = 0; i < n; i++) {
        free(A[i]);
    }
    free(A);
}

int main() {
    // Gauss begin
    double start_time_p, end_time_p; // for parallel p2
    clock_t start_time_s, end_time_s;   // for serial

    int n = 1000;

    // set no of threads
    omp_set_num_threads(NO_OF_CORES);

    // Dynamically allocate memory
    float **A = (float **)malloc(n * sizeof(float *));
    for (int i = 0; i < n; i++) {
        A[i] = (float *)malloc((n + 1) * sizeof(float));
    }

    // Initialize matrix
	initializeMatrix(A, n);

    // Serial Gaussian elimination
    start_time_s = clock();  // Start time for the program
    gaussianElimination_s(A, n);
    end_time_s = clock();
    double execution_time_s = (double)(end_time_s - start_time_s) / CLOCKS_PER_SEC;

	// Parallel Gaussina Elimination
    start_time_p = omp_get_wtime();
	gaussianElimination_s(A, n);
    end_time_p = omp_get_wtime();
    double execution_time_p = end_time_p - start_time_p;

    // Back-substitution
    float *x = (float *)malloc(n * sizeof(float));
    for (int i = n - 1; i >= 0; i--) {
        x[i] = A[i][n];
        for (int j = i + 1; j < n; j++) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }

    // Print the solution
    // printSolution(x, n);

    // print the execution time
    printf("\nSerial Execution Time (seconds): %f\n", execution_time_s);
    printf("Execution time for Parallel (seconds): %f\n\n", execution_time_p); 
    

    // Free dynamically allocated memory
    freeMemory(A, x, n);

    return 0;
}

