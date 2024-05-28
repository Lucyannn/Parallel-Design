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

void gaussElimination(float** A, int N) {
    #pragma omp parallel
    {
        #pragma omp for
        for (int k = 0; k < N; ++k) {
            // ���߳�����������
            for (int j = k + 1; j < N; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;

            // ���������̣߳�������ȥ����
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
    std::vector<int> sizes = {100, 250, 500, 750, 1000, 2000, 3000, 4000};

    std::ofstream outfile("time.csv");
    outfile << "Matrix Size, Time (seconds)" << std::endl;

    double start, end;

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

        start = omp_get_wtime();
        gaussElimination(A, N);
        end = omp_get_wtime();

        double elapsed = end - start;

        std::cout << "Matrix size: " << N << ", Time taken: " << elapsed << " seconds.\n";
        outfile << sizeIndex + 1 << ", " << N << ", " << elapsed << std::endl;

        // �����ڴ�
        for (int i = 0; i < N; i++) {
            delete[] A[i];
        }
        delete[] A;
    }

    outfile.close();
    return 0;
}
