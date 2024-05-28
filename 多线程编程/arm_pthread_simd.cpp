#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <pthread.h>
#include <sys/time.h>
#include <string>
#include <arm_neon.h>
#define MAX_THREADS 8 // 设置最大线程数

void generateMatrix(float** A, int N) {
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; i++) {
        A[i][i] = 1.0;
        for (int j = 0; j < i; j++) {
            A[i][j] = 0.0;
        }
        for (int j = i + 1; j < N; j++) {
            A[i][j] = rand()%1000;
        }
    }
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] += A[k][j];
                A[i][j]=(int)A[i][j]%1000;
            }
        }
    }
}

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

    float32x4_t vaik = vdupq_n_f32(A[i][k]);
    int j;
    for (j = k + 1; j + 4 <= n; j += 4) {
        float32x4_t vakj = vld1q_f32(&A[k][j]);
        float32x4_t vaij = vld1q_f32(&A[i][j]);
        float32x4_t vx = vmulq_f32(vakj, vaik);
        vaij = vsubq_f32(vaij, vx);
        vst1q_f32(&A[i][j], vaij);
    }
        for (j; j < n;j++) {
            A[i][j] -= A[k][j] * A[i][k];
        }
        A[i][k] = 0.0f;
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

        // 创建线程并传递参数
        for (int t_id = 0; t_id < worker_count; t_id++) {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
            param[t_id].A = A;
            param[t_id].n = N;
            pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
        }

        // 主线程挂起等待所有的工作线程完成此轮消去工作
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_join(handles[t_id], NULL);
        }
    }
}


int main() {
    std::vector<int> sizes;
    sizes.push_back(100);
    sizes.push_back(250);
    sizes.push_back(500);
    sizes.push_back(750);
    sizes.push_back(1000);
    sizes.push_back(2000);
    sizes.push_back(3000);
    sizes.push_back(4000);
    struct timeval start, end;

    for (int sizeIndex = 0; sizeIndex < sizes.size(); sizeIndex++) {
        int N = sizes[sizeIndex];
        float** A = new float*[N];
        for (int i = 0; i < N; i++) {
            A[i] = new float[N];
        }
        // 生成测试矩阵
        generateMatrix(A, N);
        //高斯消去
        gettimeofday(&start, NULL);
        gaussElimination(A,N);
        gettimeofday(&end, NULL);

        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

        std::cout << "Matrix size: " << N << ", Time taken: " << elapsed << " seconds.\n";

        // 清理内存
        for (int i = 0; i < N; i++) {
            delete[] A[i];
        }
        delete[] A;
    }

    return 0;
}

