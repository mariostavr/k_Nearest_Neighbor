#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <time.h>

FILE *fptr;
int counter = 0; //Global variable.

//================================= KNN_RESULT_FORMAT ==================================
typedef struct knnresult{
    int *nidx;      //!< Indices (0-based) of nearest neighbors [m-by-k]
    double *ndist;  //!< Distance of nearest neighbors          [m-by-k]
    int m;          //!< Number of query points                 [scalar]
    int k;          //!< Number of nearest neighbors            [scalar]
}knnresult;
struct knnresult knnR;
//======================================================================================

//============================== WRITING_ARRAY_INTO_FILE ===============================
void writeDArray(double *array, int rows, int columns, FILE *file){ // Type: Double
    int i, j;
    for (i=0; i<rows; i++){
        for (j=0; j<columns; j++)
            fprintf(file,"%f\t", *(array + i*columns + j));
        fprintf(file,"\n");
    }
}//=====================================================================================
void writeIArray(int *array, int rows, int columns, FILE *file){ // Type: Intiger
    int i, j;
    for (i=0; i<rows; i++){
        for (j=0; j<columns; j++)
            fprintf(file,"%d\t", *(array + i*columns + j));
        fprintf(file,"\n");
    }
}//=====================================================================================

//============================== SWAP - SELECTION SORT =================================
void swap(double *xp, double *yp) {
    int temp = *xp;
    *xp = *yp;
    *yp = temp;
}

void selectionSort(double *dist, int n, int k){
    int i, j, min_idx;
    for (i=0; i<k; i++){ //Sort only the first k-elements.
        min_idx = i;
        for (j=i+1; j<n; j++)
            if (dist[j] < dist[min_idx])
                min_idx = j;
        swap(&dist[min_idx], &dist[i]);
        knnR.nidx[counter] = min_idx;
        counter++;
    }
}
//=======================================================================================

//======================================= K_SELECT ======================================
void kselect(double *array, int n, int m, int k){
    int i, j, c = 0;
    double *col = (double *)malloc(n * sizeof(double *));

    for (j=0; j<m; j++){
        for (i=0; i<n; i++)
            col[i] = *(array + i*m + j); //Seperating columns of D array.
        selectionSort(col, n, k); //Using selectionSort to get the "k" minimum distances
        for (i=0; i<k; i++){
            knnR.ndist[c] = col[i];
            c++;
        }
    }
}//=====================================================================================

//====================================== COMPUTE =======================================
void kNN(double *X, double *Y, int n, int m, int d, int k){
    double *D = (double*)malloc(n*m * sizeof(double *));
    double sumX[n];
    double sumY[m];
    int i,c=0;
    for(i=0; i<n; i++)
        sumX[i] = cblas_ddot(d, X+i*d, 1, X+i*d, 1);
    for(i=0; i<m; i++)
        sumY[i] = cblas_ddot(d, Y+i*d, 1, Y+i*d, 1);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, -2, X, d, Y, d, 0, D, m);
    for (int i=0; i<n; i++)
        for (int j=0; j<m; j++){
            D[c] += sumX[i] + sumY[j];
            c++;
        }
    kselect(D, n, m, k);
}//=====================================================================================

//======================================== MAIN ========================================
int main(int argc, char** argv) {

    clock_t start, end;
    double eTime;

    int i;
    int n=20000, m=20000, d=5;
    int k=10, c=0;

    int *idx = (int*)malloc(m*k * sizeof(int *));
    knnR.nidx = idx;
    double *distances = (double*)malloc(m*k * sizeof(double *));
    knnR.ndist = distances;

    knnR.k = k;
    knnR.m = m;

    double *X = (double*)malloc(n*d * sizeof(double *));
    double *Y = (double*)malloc(m*d * sizeof(double *));

    srand(time(0)); //Generate different numbers.

    for (int i=0; i<n*d; i++)
        X[i] = rand() % 10000;

    for (int i=0; i<m*d; i++)
        Y[i] = rand() % 10000;


    start = clock();
    kNN(X, Y, n, m, d, k);
    end = clock();
    eTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("\nTime measured: %f seconds.\n", eTime);

    //Saving into files.
    fptr = fopen("Indices_V0.txt","a");
    writeIArray(knnR.nidx,m,k,fptr);
    fptr = fopen("K-Distances_V0.txt","a");
    writeDArray(knnR.ndist,m,k,fptr);
    fclose(fptr);


    printf("\n");
}
