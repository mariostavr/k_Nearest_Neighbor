#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cblas.h>
#include <time.h>
#include <mpi.h>

#define div 2000 //Divide corpus set to more parts.

FILE *fptrI, *fptrD;

int c;
int counter = 0; //Global variables
clock_t start, end;
double eTime;

//================================= KNN_RESULT_FORMAT ==================================
typedef struct knnresult{
    int *nidx;      //!< Indices (0-based) of nearest neighbors [m-by-k]
    double *ndist;  //!< Distance of nearest neighbors          [m-by-k]
    int m;          //!< Number of query points                 [scalar]
    int k;          //!< Number of nearest neighbors            [scalar]
}knnresult;
struct knnresult knn;
//======================================================================================

//============================== WRITING_ARRAY_INTO_FILE ===============================
void writeDArray(double *array, int rows, int columns, FILE *file){
    int i, j;
    for (i=0; i<rows; i++){
        for (j=0; j<columns; j++)
            fprintf(file,"%f\t", *(array + i*columns + j));
        fprintf(file, "\n");
    }
}//========================================================================================

//============================== WRITING_ARRAY_INTO_FILE ===============================
void writeIArray(int *array, int rows, int columns, FILE *file){
    int i, j;
    for (i=0; i<rows; i++){
        for (j=0; j<columns; j++)
            fprintf(file, "%d\t", *(array + i*columns + j));
        fprintf(file, "\n");
    }
}//=====================================================================================

//====================================== CONVERT D ARRAY ===================================
void convert(double *nD, double *array, int sizeA, int sizeB, int sizeD, int counter2){
    for (int i=0; i<sizeA; i++){
        for (int j=0; j<sizeB; j++){
            *(nD + i*sizeD +(counter2*sizeB+j)) = *(array + i*sizeB + j);
        }
    }
}//=======================================================================================

//========================================= SCATTERV =====================================
void divideArray(int *sendcounts, int *displs, int numProcs, int n, int d){
    int sum = 0;
    for (int i=0; i<numProcs; i++) {
        sendcounts[i] = (n*d)/numProcs;

        displs[i] = sum;
        sum += sendcounts[i];
    }
}//========================================================================================

//================================ SELECTION SORT ==============================
void swap(double *x, double *y){
    int temp = *x;
    *x = *y;
    *y = temp;
}

void selectionSort(double *row, int n, int k){
    int i, j, min_idx;
    for (i=0; i<k; i++){ //Sort only the first k-elements.
        min_idx = i;
        for (j=i+1; j<n; j++)
            if (row[j] < row[min_idx])
                min_idx = j;
        swap(&row[min_idx], &row[i]);
        knn.nidx[counter] = min_idx;
        counter++;
    }
}//============================================================================================

//======================================= K-SELECT ============================================
void kselect(double *D, int n, int size, int k){
    double *row = (double *)malloc(n * sizeof(double *));
    for (int i=0; i<size; i++){
        for (int j=0; j<n; j++)
            row[j] = *(D + i*n + j);
        selectionSort(row, n, k);
        for (int l=0; l<k; l++)
            knn.ndist[l] = row[l];
        fptrD = fopen("D.txt","a");
        writeDArray(knn.ndist,1,k, fptrD); //Writing K selected distances
    }
}//========================================================================================

//======================================== COMPUTE ========================================
void distrAllkNN(double *X, int n, int d, int rank, int numProcs, int k){

    int elements_per_process = (n*d)/numProcs; //For MPI-ScatterV - Splits Corpus Set
    int elements = (n*d)/div; //For MPI-Send and MPI-Recv - Splits Query Set

    //DIVIDE ARRAY TO PROCESS - SCATTERV
    int *sendcounts = malloc(sizeof(int)*numProcs);
    int *displs = malloc(sizeof(int)*numProcs);

    divideArray(sendcounts, displs, numProcs, n, d); //ScatterV - Initializing variables
    double rec_buf[elements_per_process];
    MPI_Scatterv(X, sendcounts, displs, MPI_DOUBLE, &rec_buf, elements_per_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int sizeA = elements_per_process/d; //Number of points (=Number of lines)
    int sizeB = elements/d;
    double *localArray = (double *)malloc(elements * sizeof(double));
    double *D = (double*)malloc(sizeA*sizeB * sizeof(double *));
    double *nD = (double*)malloc(sizeA*n * sizeof(double *));
    double sumX[sizeA], sumY[n];

    if(rank == 0)
        start = clock();

    for (int i=0; i<sizeA; i++)
        sumX[i] = cblas_ddot(d, rec_buf+i*d, 1, rec_buf+i*d, 1);

    //Sending parts of corpus set from one processor to the next processor
    for(int i=0; i<div; i++){
        MPI_Send(&X[i*elements], elements, MPI_DOUBLE, (rank+1)%numProcs, 0, MPI_COMM_WORLD);
    }
    //Receiving - Computing D Array
    for (int i=0; i<div; i++){
        MPI_Recv(localArray, elements, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for(int j=0; j<sizeB; j++)
            sumY[j] = cblas_ddot(d, localArray+j*d, 1, localArray+j*d, 1);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, sizeA, sizeB, d, -2, rec_buf, d, localArray, d, 0, D, sizeB);
        c=0;
        for(int l=0; l<sizeA; l++)
            for (int q=0; q<sizeB; q++){
                D[c] += sumX[l] + sumY[q];
                c++;
            }
        convert(nD,D,sizeA,sizeB,n,i); //Explained in report.
    }
    kselect(nD, n, sizeA, k);
    fptrI = fopen("I.txt","a");
    writeIArray(knn.nidx,sizeA,k, fptrI); //Writing indices in text file.

    if(rank == numProcs-1){
        end = clock();
        eTime = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("\nTime measured: %f seconds.\n\n", eTime);
    }


}//==============================================================================================


//============================================= MAIN ============================================
int main(int argc, char *argv[]){

    int n=30000, d=10;
    int k=10;

    int *idx = (int*)malloc(n*k * sizeof(int *));
    knn.nidx = idx;
    double *distances = (double*)malloc(n * sizeof(double *));
    knn.ndist = distances;
    double *X = (double*)malloc(n*d * sizeof(double *));

    srand(time(0));
    for (int i=0; i<n*d; i++)
        X[i] = rand()%10000;

    int rank, numProcs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    distrAllkNN(X,n,d,rank,numProcs,k);

    MPI_Finalize();
}
