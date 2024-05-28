#include <iostream>
#include <fstream>
#include <vector>
#include <pthread.h>
#include <windows.h>
#include <immintrin.h> // ���� AVX ���ͷ�ļ�
#define MAX_THREADS 16 // ��������߳���

typedef struct {
    int k;      // ��ȥ���ִ�
    int t_id;   // �߳� id
    float** A;  // ����
    int n;      // �����С
} threadParam_t;

void* threadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;           // ��ȥ���ִ�
    int t_id = p->t_id;     // �̱߳��
    float** A = p->A;       // ����
    int n = p->n;           // �����С

    int i = k + t_id + 1;   // ��ȡ�Լ��ļ�������

    int avx_count = (n - k - 1) / 8;  // ������Դ���� AVX ��������
    int remainder = (n - k - 1) % 8;  // �������һ���������� AVX �����Ĵ�С

    // ʹ�� AVX �������в��м���
    for (int j = k + 1; j < k + 1 + avx_count * 8; j += 8) {
        __m256 row_i = _mm256_loadu_ps(&A[i][j]);
        __m256 row_k = _mm256_loadu_ps(&A[k][j]);
        __m256 product = _mm256_mul_ps(row_k, _mm256_set1_ps(A[i][k]));
        row_i = _mm256_sub_ps(row_i, product);
        _mm256_storeu_ps(&A[i][j], row_i);
    }

    // �������һ���������� AVX ����
    for (int j = k + 1 + avx_count * 8; j < n; ++j) {
        A[i][j] -= A[i][k] * A[k][j];
    }

    A[i][k] = 0;
    pthread_exit(NULL);
}

void gaussElimination(float** A, int N) {
    for (int k = 0; k < N; ++k) {
        // ���߳�����������
        for (int j = k + 1; j < N; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        // ���������߳�,������ȥ����
        int worker_count = N - 1 - k; // �����߳�����
        pthread_t handles[MAX_THREADS];
        threadParam_t param[MAX_THREADS];
        // ��������
        for (int t_id = 0; t_id < worker_count; t_id++) {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
            param[t_id].A = A;
            param[t_id].n = N;
        }
        // �����߳�
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_create(&handles[t_id], NULL, threadFunc, (void *)&param[t_id]);
        }

        // ���̹߳���ȴ����еĹ����߳���ɴ�����ȥ����
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_join(handles[t_id], NULL);
        }
    }
    return;
}


int main() {
    std::vector<int> sizes = {100, 250, 500, 750, 1000, 2000, 3000, 4000};

    std::ofstream outfile("time.csv");
    outfile << "Matrix Size, Time (seconds)" << std::endl;

    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);

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

        QueryPerformanceCounter(&start);
        // ʹ�� simd ���л��ĸ�˹��ȥ����
        gaussElimination(A, N);
        QueryPerformanceCounter(&end);

        double elapsed = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;

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
