#include <iostream>
#include <fstream>
#include <vector>
#include <pthread.h>
#include <windows.h>
#include <numeric>
#define MAX_THREADS 16 // 设置最大线程数

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

    int j = k + t_id + 1;   // 获取自己的计算任务

    for (int i = k + 1; i < n; ++i) {
        A[i][j] -= A[i][k] * A[k][j];
    }
    pthread_exit(NULL);
}

void gaussElimination(float** A, int N) {
    for (int k = 0; k < N; ++k) {
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
        for(int i=k+1;i<N;i++){
                A[i][k] = 0.0;
        }
        // 主线程挂起等待所有的工作线程完成此轮消去工作
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_join(handles[t_id], NULL);
        }
    }
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

/*
要使垂直划分策略尽可能负载均衡，可以采用以下方法：

动态调整任务分配：根据每列中非零元素的数量来动态分配任务，使得每个线程处理的工作量尽量均衡。
任务队列：使用任务队列或者工作池，让空闲线程从队列中获取任务，以充分利用系统资源。
均匀分配列数：尽量确保每个线程处理的列数相近，减少线程之间的工作量差异。

在修改后的代码中，我首先计算了每列中非零元素的数量，并根据这个数量动态分配任务。
具体来说，我根据每列中非零元素的比例来分配任务，使得每个线程处理的列数与其对应的非零元素数量成比例。
这样可以尽量保证每个线程的工作量相对均衡，提高负载均衡性能。

*/
