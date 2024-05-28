#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>
#include <immintrin.h>
typedef struct {
    int k;      // 消去的轮次
    int t_id;   // 线程 id
    float** A;  // 矩阵
    int n;      // 矩阵大小
} threadParam_t;

void threadFunc(threadParam_t* param) {
    int k = param->k;           // 消去的轮次
    int t_id = param->t_id;     // 线程编号
    float** A = param->A;       // 矩阵
    int n = param->n;           // 矩阵大小

    int i = k + t_id + 1;   // 获取自己的计算任务

    int avx_count = (n - k - 1) / 8;  // 计算可以处理的 AVX 向量数量
    int remainder = (n - k - 1) % 8;  // 计算最后一个不完整的 AVX 向量的大小

    // 使用 AVX 向量进行并行计算
    for (int j = k + 1; j < k + 1 + avx_count * 8; j += 8) {
        __m256 row_i = _mm256_loadu_ps(&A[i][j]);
        __m256 row_k = _mm256_loadu_ps(&A[k][j]);
        __m256 product = _mm256_mul_ps(row_k, _mm256_set1_ps(A[i][k]));
        row_i = _mm256_sub_ps(row_i, product);
        _mm256_storeu_ps(&A[i][j], row_i);
    }
    // 处理最后一个不完整的 AVX 向量
    for (int j = k + 1 + avx_count * 8; j < n; ++j) {
        A[i][j] -= A[i][k] * A[k][j];
    }

    A[i][k] = 0;
}


void gaussElimination(float** A, int N) {
    #pragma omp parallel
    {
        #pragma omp for
        for (int k = 0; k < N; ++k) {
            // 主线程做除法操作
            for (int j = k + 1; j < N; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
            // 创建工作线程，进行消去操作
            int worker_count = N - 1 - k; // 工作线程数量
            threadParam_t param[N];
            #pragma omp parallel for
            for (int t_id = 0; t_id < worker_count; t_id++) {
                param[t_id].k = k;
                param[t_id].t_id = t_id;
                param[t_id].A = A;
                param[t_id].n = N;
            }

            // 开启并行区域
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

        // 从文件中读取矩阵数据
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

        // 清理内存
        for (int i = 0; i < N; i++) {
            delete[] A[i];
        }
        delete[] A;
    }

    outfile.close();
    return 0;
}
