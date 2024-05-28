#include <iostream>
#include <fstream>
#include <vector>
#include <pthread.h>
#include <windows.h>
#define MAX_THREADS 12 // 设置最大线程数

typedef struct {
    int k;      // 消去的轮次
    int t_id;   // 线程 id
    float** A;  // 矩阵
    int n;      // 矩阵大小
} threadParam_t;

void* threadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;           // 消去的轮次
    int t_id = p->t_id;     // 线程编号
    float** A = p->A;       // 矩阵
    int n = p->n;           // 矩阵大小

    int i = k + t_id + 1;   // 获取自己的计算任务

    for (int j = k + 1; j < n; ++j) {
        A[i][j] -= A[i][k] * A[k][j];
    }
    A[i][k] = 0;

    pthread_exit(NULL);
}

void gaussElimination(float** A, int N) {
    for (int k = 0; k < N; k++) {
        // 主线程做除法操作
        for (int j = k + 1; j < N; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;

        // 创建工作线程，进行消去操作
        int worker_count = N - 1 - k; // 工作线程数量
        pthread_t handles[MAX_THREADS];
        threadParam_t param[MAX_THREADS];
        // 分配任务
        for (int t_id = 0; t_id < worker_count; t_id++) {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
            param[t_id].A = A;
            param[t_id].n = N;
        }
        // 创建线程
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
        }

        // 主线程挂起等待所有的工作线程完成此轮消去工作
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_join(handles[t_id], NULL);
        }
    }
}

int main() {
    std::vector<int> sizes = {100, 250, 500, 750, 1000, 2000, 3000, 4000};

    std::ofstream outfile("time-12.csv");
    outfile << "Matrix Size, Time (seconds)" << std::endl;

    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);

    for (int sizeIndex = 0; sizeIndex < sizes.size(); ++sizeIndex) {
        int N = sizes[sizeIndex];
        float** A = new float*[N];
        for (int i = 0; i < N; i++) {
            A[i] = new float[N];
        }

        // 从文件中读取矩阵数据
        std::ifstream infile("matrix" + std::to_string(sizeIndex + 1) + ".txt");
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                infile >> A[i][j];
            }
        }
        infile.close();

        QueryPerformanceCounter(&start);
        gaussElimination(A, N);
        QueryPerformanceCounter(&end);

        double elapsed = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;

        std::cout << "Matrix size: " << N << ", Time taken: " << elapsed << " seconds.\n";
        outfile << sizeIndex + 1 << ", " << N << ", " << elapsed << std::endl;

        // 清理内存
        for (int i = 0; i < N; i++) {
            delete[] A[i];
        }
        delete[] A;
    }

    outfile.close();
    return 0;
}
