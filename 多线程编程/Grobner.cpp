#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <chrono>
#include <windows.h>
#include <emmintrin.h>  //SSE2
#include <immintrin.h>  //AVX

using namespace std;

// ����ȫ�ֱ������洢��Ԫ��ӳ��
unordered_map<int, vector<int>*> iToBasis;

// ��ϡ������תΪʵ�ʾ���
vector<int> denseToActual(const vector<int>& denseRow, int size) {
    vector<int> actualRow(size, 0);
    for (int idx : denseRow) {
        if (idx < size) {
            actualRow[idx] = 1;
        }
    }
    return actualRow;
}

// ��ʵ�ʾ���תΪϡ������
vector<int> actualToDense(const vector<int>& actualRow) {
    vector<int> denseRow;
    for (int i = 0; i < actualRow.size(); i++) {
        if (actualRow[i] != 0) {
            denseRow.push_back(i);
        }
    }
    return denseRow;
}

// ���ļ��ж�ȡ��Ԫ�Ӻͱ���Ԫ��
bool readData(vector<vector<int>>& eliminators, vector<vector<int>>& eliminated, const string& eliminator_file, const string& eliminated_file, int size) {
    ifstream eliminator_stream(eliminator_file);
    ifstream eliminated_stream(eliminated_file);

    if (!eliminator_stream.is_open() || !eliminated_stream.is_open()) {
        cout << "Failed to open file(s)." << endl;
        return false;
    }

    string line;
    while (getline(eliminator_stream, line)) {
        istringstream iss(line);
        vector<int> row;
        int element;
        while (iss >> element) {
            row.push_back(element);
        }
        eliminators.push_back(denseToActual(row, size));
    }

    while (getline(eliminated_stream, line)) {
        istringstream iss(line);
        vector<int> row;
        int element;
        while (iss >> element) {
            row.push_back(element);
        }
        eliminated.push_back(denseToActual(row, size));
    }

    eliminator_stream.close();
    eliminated_stream.close();
    return true;
}

// �����е��׸�����λ������
void updateLeadingOne(vector<int>& row, int& leadingOneIndex) {
    leadingOneIndex = -1;
    for (int i = 0; i < row.size(); i++) {
        if (row[i] == 1) {
            leadingOneIndex = i;
            break;
        }
    }
}

//�����㷨-------------------------------------------------------------------------------------------------------------
void gaussianEliminationBatch(vector<vector<int>>& eliminators, vector<vector<int>>& eliminated) {
    vector<vector<int>> newEliminators;

    for (auto& row : eliminated) {
        int leadingOneIndex = -1;
        updateLeadingOne(row, leadingOneIndex);  // �����׸�����Ԫ�ص�λ��

        while (leadingOneIndex != -1) {
            auto it = iToBasis.find(leadingOneIndex);
            if (it != iToBasis.end()) {
                // �ҵ���Ԫ�ӣ�����������
                bool isChanged = false; // ���λ�仯
                for (int i = 0; i < row.size(); i++) {
                    int oldValue = row[i];
                    row[i] ^= it->second->at(i);
                    if (oldValue != row[i]) {
                        isChanged = true; // ����һ��λ�����仯
                    }
                }
                if (!isChanged) {
                    // ���û���κθı䣬��������ѭ��
                    break;
                }
                updateLeadingOne(row, leadingOneIndex);  // �ٴθ����׸�����Ԫ�ص�λ��
            } else {
                // ����ǰ������Ϊ�µ���Ԫ��
                iToBasis[leadingOneIndex] = &row;
                newEliminators.push_back(row);
                break;  // �˳�ѭ��
            }
        }

        // ������Ƿ�ȫΪ0
        if (leadingOneIndex == -1) {
            row.clear();  // �����
        }
    }

    // ������Ԫ�Ӽ��뵽������Ԫ���б���
    for (auto& newElim : newEliminators) {
        eliminators.push_back(newElim);
    }
}
//SSE-------------------------------------------------------------------------------------------------------------------------

void gaussianEliminationBatch_SSE(vector<vector<int>>& eliminators, vector<vector<int>>& eliminated) {
    vector<vector<int>> newEliminators;

    for (auto& row : eliminated) {
        int leadingOneIndex = -1;
        updateLeadingOne(row, leadingOneIndex);  // �����׸�����Ԫ�ص�λ��

        while (leadingOneIndex != -1) {
            auto it = iToBasis.find(leadingOneIndex);
            if (it != iToBasis.end()) {
                // ʹ�� SSE ����������
                __m128i row_vec = _mm_setzero_si128();  // ��ʼ�� row ����
                __m128i elim_vec = _mm_loadu_si128((__m128i*)it->second->data());  // ������Ԫ������
                bool isChanged = false;

                for (int i = 0; i < row.size(); i += 4) {
                    // ���� row ����
                    row_vec = _mm_loadu_si128((__m128i*)&row[i]);

                    // ִ��������
                    row_vec = _mm_xor_si128(row_vec, elim_vec);

                    // �洢�޸ĺ�� row ����
                    _mm_storeu_si128((__m128i*)&row[i], row_vec);

                    // ����Ƿ����κ�λ�����仯
                    __m128i diff = _mm_xor_si128(_mm_loadu_si128((__m128i*)&row[i]), _mm_loadu_si128((__m128i*)&oldRow[i]));
                    if (_mm_movemask_epi8(diff) != 0) {
                        isChanged = true;
                    }
                }

                if (!isChanged) {
                    // ���û���κθı䣬��������ѭ��
                    break;
                }
                updateLeadingOne(row, leadingOneIndex);  // �ٴθ����׸�����Ԫ�ص�λ��
            } else {
                // ����ǰ������Ϊ�µ���Ԫ��
                iToBasis[leadingOneIndex] = &row;
                newEliminators.push_back(row);
                break;  // �˳�ѭ��
            }
        }

        // ������Ƿ�ȫΪ 0
        if (leadingOneIndex == -1) {
            row.clear();  // �����
        }
    }

    // ������Ԫ�Ӽ��뵽������Ԫ���б���
    for (auto& newElim : newEliminators) {
        eliminators.push_back(newElim);
    }
}

//AVX----------------------------------------------------------------------------------------------------------------------------------
void gaussianEliminationBatch_AVX(vector<vector<int>>& eliminators, vector<vector<int>>& eliminated) {
    vector<vector<int>> newEliminators;

    for (auto& row : eliminated) {
        int leadingOneIndex = -1;
        updateLeadingOne(row, leadingOneIndex);  // �����׸�����Ԫ�ص�λ��

        while (leadingOneIndex != -1) {
            auto it = iToBasis.find(leadingOneIndex);
            if (it != iToBasis.end()) {
                // ʹ�� AVX ����������
                __m256i row_vec = _mm256_setzero_si256();  // ��ʼ�� row ����
                __m256i elim_vec = _mm256_loadu_si256((__m256i*)it->second->data());  // ������Ԫ������
                bool isChanged = false;

                for (int i = 0; i < row.size(); i += 8) {
                    // ���� row ����
                    row_vec = _mm256_loadu_si256((__m256i*)&row[i]);

                    // ִ��������
                    row_vec = _mm256_xor_si256(row_vec, elim_vec);

                    // �洢�޸ĺ�� row ����
                    _mm256_storeu_si256((__m256i*)&row[i], row_vec);

                    // ����Ƿ����κ�λ�����仯
                    __m256i diff = _mm256_xor_si256(_mm256_loadu_si256((__m256i*)&row[i]), _mm256_loadu_si256((__m256i*)&oldRow[i]));
                    if (_mm256_movemask_epi8(diff) != 0) {
                        isChanged = true;
                    }
                }

                if (!isChanged) {
                    // ���û���κθı�,��������ѭ��
                    break;
                }
                updateLeadingOne(row, leadingOneIndex);  // �ٴθ����׸�����Ԫ�ص�λ��
            } else {
                // ����ǰ������Ϊ�µ���Ԫ��
                iToBasis[leadingOneIndex] = &row;
                newEliminators.push_back(row);
                break;  // �˳�ѭ��
            }
        }

        // ������Ƿ�ȫΪ 0
        if (leadingOneIndex == -1) {
            row.clear();  // �����
        }
    }

    // ������Ԫ�Ӽ��뵽������Ԫ���б���
    for (auto& newElim : newEliminators) {
        eliminators.push_back(newElim);
    }
}



// �����������β����ɽ��
void processGaussianElimination(vector<vector<int>>& initialEliminators, vector<vector<int>>& allRows, vector<vector<int>>& result) {
    // ���� initialEliminators �� allRows �Ѱ����ηָ��
    for (size_t batchStart = 0; batchStart < allRows.size(); batchStart += 10) {  // ÿ�� 10 ��
        size_t batchEnd = std::min(batchStart + 10, allRows.size());
        vector<vector<int>> batchRows(allRows.begin() + batchStart, allRows.begin() + batchEnd);

        gaussianEliminationBatch(initialEliminators, batchRows);
        //gaussianEliminationBatch_SSE(initialEliminators, batchRows);
        //gaussianEliminationBatch_AVX(initialEliminators, batchRows);

        // �ռ��ǿ���
        for (auto& row : batchRows) {
            if (!row.empty()) {
                result.push_back(row);
            }
        }
    }

    // ����Ԫ�����ӵ����
    for (const auto& eliminator : initialEliminators) {
        result.push_back(actualToDense(eliminator));
    }
}

// ���������д���ļ�
bool writeResult(const vector<vector<int>>& result, const string& output_file) {
    ofstream output_stream(output_file);

    if (!output_stream.is_open()) {
        cout << "Failed to open output file." << endl;
        return false;
    }

    for (const auto& row : result) {
        for (int element : row) {
            output_stream << element << " ";
            cout<<element<<" ";
        }
        output_stream << endl;
        cout<<endl;
    }
    output_stream.close();
    return true;
}

//////////////////////////////////////////////////////  pthread /////////////////////////////////////////////

#include <pthread.h>
#include <vector>
#include <algorithm>

using namespace std;

// �����̲߳����ṹ��
struct ThreadParam {
    vector<vector<int>>& eliminators;
    vector<vector<int>>& rows;
    vector<vector<int>>& result;
    size_t start;
    size_t end;
};

// ��˹��Ԫ����������
void* gaussianEliminationBatchThread(void* arg) {
    ThreadParam* param = (ThreadParam*)arg;

    for (size_t i = param->start; i < param->end; ++i) {
        vector<vector<int>> newEliminators;
        vector<int>& row = param->rows[i];

        // ��˹��Ԫ����
        // ʡ�Բ��ִ��룬����ʹ����㷨��ͬ

        // ������Ԫ�Ӽ��뵽������Ԫ���б���
        for (auto& newElim : newEliminators) {
            param->eliminators.push_back(newElim);
        }
    }

    pthread_exit(NULL);
}

// �����������β����ɽ��
void processGaussianEliminationpthread(vector<vector<int>>& initialEliminators, vector<vector<int>>& allRows, vector<vector<int>>& result) {
    size_t totalRows = allRows.size();
    size_t batchSize = min(totalRows, (size_t)10); // ÿ�� 10 ��

    // �����߳̾������
    pthread_t threads[16];
    // �����̲߳�������
    ThreadParam params[16];

    size_t start = 0;
    size_t numThreads = min((totalRows + batchSize - 1) / batchSize, (size_t)16); // ���16���߳�

    for (size_t i = 0; i < numThreads; ++i) {
        size_t end = min(start + batchSize, totalRows);
        params[i] = {initialEliminators, allRows, result, start, end};

        // �����߳�
        pthread_create(&threads[i], NULL, gaussianEliminationBatchThread, (void*)&params[i]);

        start = end;
    }

    // �ȴ������߳����
    for (size_t i = 0; i < numThreads; ++i) {
        pthread_join(threads[i], NULL);
    }

    // ����Ԫ�����ӵ����
    for (const auto& eliminator : initialEliminators) {
        result.push_back(eliminator);
    }
}
/////////////////////////////////////////////OpenMP////////////////////////////////////////////
#include <omp.h>
#include <vector>
#include <algorithm>

using namespace std;

// ��˹��Ԫ����������
void gaussianEliminationBatch(vector<vector<int>>& eliminators, vector<vector<int>>& rows) {
    vector<vector<int>> newEliminators;

    #pragma omp parallel for shared(eliminators, rows)
    for (size_t i = 0; i < rows.size(); ++i) {
        vector<int>& row = rows[i];
        int leadingOneIndex = -1;

        // �����׸�����Ԫ�ص�λ��
        updateLeadingOne(row, leadingOneIndex);

        while (leadingOneIndex != -1) {
            auto it = iToBasis.find(leadingOneIndex);
            if (it != iToBasis.end()) {
                // �ҵ���Ԫ�ӣ�����������
                bool isChanged = false; // ���λ�仯
                for (int j = 0; j < row.size(); j++) {
                    int oldValue = row[j];
                    row[j] ^= it->second->at(j);
                    if (oldValue != row[j]) {
                        isChanged = true; // ����һ��λ�����仯
                    }
                }
                if (!isChanged) {
                    // ���û���κθı䣬��������ѭ��
                    break;
                }
                // �ٴθ����׸�����Ԫ�ص�λ��
                updateLeadingOne(row, leadingOneIndex);
            } else {
                // ����ǰ������Ϊ�µ���Ԫ��
                #pragma omp critical
                {
                    iToBasis[leadingOneIndex] = &row;
                    newEliminators.push_back(row);
                }
                break;  // �˳�ѭ��
            }
        }

        // ������Ƿ�ȫΪ0
        if (leadingOneIndex == -1) {
            row.clear();  // �����
        }
    }

    // ������Ԫ�Ӽ��뵽������Ԫ���б���
    for (auto& newElim : newEliminators) {
        eliminators.push_back(newElim);
    }
}

// �����������β����ɽ��
void processGaussianEliminationOpenMP(vector<vector<int>>& initialEliminators, vector<vector<int>>& allRows, vector<vector<int>>& result) {
    size_t totalRows = allRows.size();
    size_t batchSize = min(totalRows, (size_t)10); // ÿ�� 10 ��

    // OpenMP ����ִ��ÿ�����εĸ�˹��Ԫ����
    #pragma omp parallel for
    for (size_t batchStart = 0; batchStart < totalRows; batchStart += batchSize) {
        size_t batchEnd = min(batchStart + batchSize, totalRows);
        vector<vector<int>> batchRows(allRows.begin() + batchStart, allRows.begin() + batchEnd);

        gaussianEliminationBatch(initialEliminators, batchRows);

        // �ռ��ǿ���
        #pragma omp critical
        {
            for (auto& row : batchRows) {
                if (!row.empty()) {
                    result.push_back(row);
                }
            }
        }
    }

    // ����Ԫ�����ӵ����
    #pragma omp parallel for
    for (const auto& eliminator : initialEliminators) {
        #pragma omp critical
        {
            result.push_back(actualToDense(eliminator));
        }
    }
}
/////////////////////////////////////// main ///////////////////////////////////////////////////////
int main() {
    vector<vector<int>> eliminators, eliminated, result;
    int matrix_size = 130;  // �����С

    // ���ļ���ȡ��Ԫ�Ӻͱ���Ԫ��
    if (!readData(eliminators, eliminated, "D:\\Parallel\\Gro_chuan\\er_1.txt", "D:\\Parallel\\Gro_chuan\\ed_1.txt", matrix_size)) {
        return 1;
    }

    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);

    LARGE_INTEGER start, end;
    QueryPerformanceCounter(&start);

    // ʹ�����δ�����˹��ȥ
    processGaussianElimination(eliminators, eliminated, result);

    QueryPerformanceCounter(&end);
    double elapsed = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;

    // �������ʱ��
    cout << "Time taken: " << elapsed << " seconds." << endl;

    // �����д���ļ�
    if (!writeResult(result, "./results/result_1.txt")) {
        return 1;
    }

    return 0;
}


