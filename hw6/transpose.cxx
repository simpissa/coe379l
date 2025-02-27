#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>

using namespace std;

void localTranspose(vector<int>& matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            swap(matrix[i * size + j], matrix[j * size + i]);
        }
    }
}

void printMatrix(const vector<int>& matrix, int size, const string& label) {
    cout << label << endl;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            cout << matrix[i * size + j] << "\t";
        }
        cout << endl;
    }
    cout << endl;
}

void divideMatrix(const vector<int>& matrix, int size, 
                 vector<int>& q00, vector<int>& q01, 
                 vector<int>& q10, vector<int>& q11) {
    int half = size / 2;
    
    q00.resize(half * half);
    q01.resize(half * half);
    q10.resize(half * half);
    q11.resize(half * half);
    
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            q00[i * half + j] = matrix[i * size + j];
            q01[i * half + j] = matrix[i * size + (j + half)];
            q10[i * half + j] = matrix[(i + half) * size + j];
            q11[i * half + j] = matrix[(i + half) * size + (j + half)];
        }
    }
}

void combineMatrix(vector<int>& matrix, int size,
                  const vector<int>& q00, const vector<int>& q01,
                  const vector<int>& q10, const vector<int>& q11) {
    int half = size / 2;
    
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            matrix[i * size + j] = q00[i * half + j];
            
            matrix[i * size + (j + half)] = q01[i * half + j];
            
            matrix[(i + half) * size + j] = q10[i * half + j];
            
            matrix[(i + half) * size + (j + half)] = q11[i * half + j];
        }
    }
}

void recursiveTranspose(vector<int>& matrix, int size, MPI_Comm comm) {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    
    if (nprocs == 1) {
        localTranspose(matrix, size);
        return;
    }
    
    if (nprocs != 4) {
        if (rank == 0) {
            cerr << "must have 4 procs" << endl;
        }
        MPI_Abort(comm, 1);
        return;
    }
    
    int half_size = size / 2;
    
    vector<int> local_quadrant(half_size * half_size);
    
    if (rank == 0) {
        vector<int> q00(half_size * half_size);
        vector<int> q01(half_size * half_size);
        vector<int> q10(half_size * half_size);
        vector<int> q11(half_size * half_size);
        
        divideMatrix(matrix, size, q00, q01, q10, q11);
        
        local_quadrant = q00;
        
        MPI_Send(q01.data(), half_size * half_size, MPI_INT, 1, 0, comm);
        
        MPI_Send(q10.data(), half_size * half_size, MPI_INT, 2, 0, comm);
        
        MPI_Send(q11.data(), half_size * half_size, MPI_INT, 3, 0, comm);
    } else {
        MPI_Recv(local_quadrant.data(), half_size * half_size, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
    }
    
    MPI_Comm sub_comm;
    MPI_Comm_split(comm, rank, 0, &sub_comm);
    
    recursiveTranspose(local_quadrant, half_size, sub_comm);
    
    MPI_Comm_free(&sub_comm);
    
    if (rank == 1) {
        MPI_Sendrecv_replace(local_quadrant.data(), half_size * half_size, MPI_INT,
                            2, 0, 2, 0, comm, MPI_STATUS_IGNORE);
    } else if (rank == 2) {
        MPI_Sendrecv_replace(local_quadrant.data(), half_size * half_size, MPI_INT,
                            1, 0, 1, 0, comm, MPI_STATUS_IGNORE);
    }
    
    if (rank == 0) {
        vector<int> q00 = local_quadrant;
        vector<int> q01(half_size * half_size);
        vector<int> q10(half_size * half_size);
        vector<int> q11(half_size * half_size);
        
        MPI_Recv(q01.data(), half_size * half_size, MPI_INT, 1, 0, comm, MPI_STATUS_IGNORE);
        
        MPI_Recv(q10.data(), half_size * half_size, MPI_INT, 2, 0, comm, MPI_STATUS_IGNORE);
        
        MPI_Recv(q11.data(), half_size * half_size, MPI_INT, 3, 0, comm, MPI_STATUS_IGNORE);
        
        combineMatrix(matrix, size, q00, q01, q10, q11);
    } else {
        MPI_Send(local_quadrant.data(), half_size * half_size, MPI_INT, 0, 0, comm);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    int N = 16;
    
    vector<int> matrix;
    
    // Initialize matrix on rank 0
    if (rank == 0) {
        matrix.resize(N * N);
        
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i * N + j] = i * N + j + 1;
            }
        }
        
        printMatrix(matrix, N, "Original Matrix:");
    }
    
	
    double t;
    t = MPI_Wtime();
        MPI_Comm quad_comm;
        int color = (rank < 4) ? 0 : 1;
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, &quad_comm);
        
        if (color == 0) {
            if (rank == 0) {
                recursiveTranspose(matrix, N, quad_comm);
                
                printMatrix(matrix, N, "Transposed Matrix:");
            } else {
                vector<int> temp;
                recursiveTranspose(temp, N, quad_comm);
            }
        }
        
        MPI_Comm_free(&quad_comm);

    t = MPI_Wtime()-t;
    
    if (rank == 0) cout<<"\nTime: "<<t;

    MPI_Finalize();
    return 0;
}
