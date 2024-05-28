#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>

typedef struct {
    int k;      // ��ȥ���ִ�
    int t_id;   // �߳� id
    float** A;  // ����
    int n;      // �����С
} threadParam_t;

void threadFunc(threadParam_t* param) {
    int k = param->k;           // ��ȥ���ִ�
    int t_id = param->t_id;     // �̱߳��
    float** A = param->A;       // ����
    int n = param->n;           // �����С

    int i = k + t_id + 1;   // ��ȡ�Լ��ļ�������

    for (int j = k + 1; j < n; ++j) {
        A[i][j] -= A[i][k] * A[k][j];
    }
    A[i][k] = 0;
}

void gaussElimination(float** A, int N, int chunk_size) {
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, chunk_size)
        for (int k = 0; k < N; ++k) {
            // ���߳�����������
            for (int j = k + 1; j < N; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;

            // ���������߳�,������ȥ����
            int worker_count = N - 1 - k; // �����߳�����
            threadParam_t param[N];
            #pragma omp parallel for
            for (int t_id = 0; t_id < worker_count; t_id++) {
                param[t_id].k = k;
                param[t_id].t_id = t_id;
                param[t_id].A = A;
                param[t_id].n = N;
            }

            // ������������
            #pragma omp parallel for
            for (int t_id = 0; t_id < worker_count; t_id++) {
                threadFunc(&param[t_id]);
            }
        }
    }
}

int main() {
    std::vector<int> sizes = {1000, 2000, 3000, 4000};
    std::vector<int> chunk_sizes = {1, 10, 50,100,150,200};

    std::ofstream outfile("time_150.csv");
    outfile << "Matrix Size, Chunk Size, Time (seconds)" << std::endl;

    for (int sizeIndex = 0; sizeIndex < sizes.size(); ++sizeIndex) {
        int N = sizes[sizeIndex];
        float** A = new float*[N];
        for (int i = 0; i < N; i++) {
            A[i] = new float[N];
        }

        // ���ļ��ж�ȡ��������
        std::ifstream infile("matrix" + std::to_string(sizeIndex + 1) + ".txt");
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                infile >> A[i][j];
            }
        }
        infile.close();

       // for (int chunkIndex = 0; chunkIndex < chunk_sizes.size(); ++chunkIndex) {
           // int chunk_size = chunk_sizes[chunkIndex];
           int chunk_size=150;
            double start = omp_get_wtime();
            gaussElimination(A, N, chunk_size);
            double end = omp_get_wtime();

            double elapsed = end - start;

            std::cout << "Matrix size: " << N << ", Chunk size: " << chunk_size << ", Time taken: " << elapsed << " seconds." << std::endl;
            outfile << N << ", " << chunk_size << ", " << elapsed << std::endl;
      // }

        // �����ڴ�
        for (int i = 0; i < N; i++) {
            delete[] A[i];
        }
        delete[] A;
    }

    outfile.close();
    return 0;
}
