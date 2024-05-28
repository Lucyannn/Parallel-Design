#include <fstream>
#include<iostream>
#include <vector>
#include <pthread.h>
#include <windows.h>
// �궨�� ��ѡ
#define MAX_THREADS_4 4
#define MAX_THREADS_8 8
#define MAX_THREADS_12 12
#define MAX_THREADS_16 16

using namespace std;
//���ݽṹ
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

    for (int j = k + 1; j < n; ++j) {
        A[i][j] -= A[i][k] * A[k][j];
    }
    A[i][k] = 0;

    pthread_exit(NULL);
}

void gaussElimination(float** A, int N, int max_threads) {
    for (int k = 0; k < N; ++k) {
        // ���߳�����������
        for (int j = k + 1; j < N; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;

        // ���������̣߳�������ȥ����
        int worker_count = min(max_threads, N - 1 - k); // �����߳�����
        //int worker_count=N-1-k;
        pthread_t handles[worker_count];
        threadParam_t param[worker_count];

        // ��������
        for (int t_id = 0; t_id < worker_count; t_id++) {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
            param[t_id].A = A;
            param[t_id].n = N;
        }

        // �����߳�
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
        }

        // ���̹߳���ȴ����еĹ����߳���ɴ�����ȥ����
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_join(handles[t_id], NULL);
        }
    }
}

int main() {
    vector<int> sizes = {100, 250, 500, 750, 1000, 2000, 3000, 4000};

    ofstream outfile("time_16.csv");
    outfile << "Matrix Size, Time (seconds)" << endl;

    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);

    for (int sizeIndex = 0; sizeIndex < sizes.size(); ++sizeIndex) {
        int N = sizes[sizeIndex];
        float** A = new float*[N];
        for (int i = 0; i < N; i++) {
            A[i] = new float[N];
        }

        // ���ļ��ж�ȡ��������
        ifstream infile("matrix" + to_string(sizeIndex + 1) + ".txt");
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                infile >> A[i][j];
            }
        }
        infile.close();

        QueryPerformanceCounter(&start);
        gaussElimination(A, N, MAX_THREADS_16); // �޸�Ϊ��ͬ������߳���
        QueryPerformanceCounter(&end);

        double elapsed = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;

        cout << "Matrix size: " << N << ", Time taken: " << elapsed << " seconds.\n";
        outfile << "4 Threads, " << sizeIndex + 1 << ", " << N << ", " << elapsed << endl;

        // �����ڴ�
        for (int i = 0; i < N; i++) {
            delete[] A[i];
        }
        delete[] A;
    }

    outfile.close();
    return 0;
}

