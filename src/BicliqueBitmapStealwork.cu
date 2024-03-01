#include<iostream>
#include<vector>
#include<map>
#include<algorithm>
#include<chrono>
#include<time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MAXSTACKSIZE 8 //max stack size
#define MAXHSIZE 60000
#define MAXSSIZE 60000

#define MAXBATCHLEVELSIZE 64
#define MAXSBATCHSIZE 2096
#define MAXHBATCHSIZE 1048

#define BLOCKNUM 128
#define THREADNUM 64

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

class Vertex {
public:
	unsigned label;
	std::vector<unsigned> neighbor;
	Vertex() {}
	Vertex(unsigned lb) {
		label = lb;
	}
};

class CSR {
public:
	unsigned* row_offset = nullptr;
	unsigned* column_index = nullptr;
	CSR() {
		row_offset = nullptr;
		column_index = nullptr;
	}
	~CSR() {
		delete[] row_offset;
		delete[] column_index;
	}
};

class BitMap {
public:
	unsigned* id_offset = nullptr;
	unsigned* notzero_id = nullptr;
	unsigned* bitcode = nullptr;
	BitMap() {
		id_offset = nullptr;
		notzero_id = nullptr;
		bitcode = nullptr;
	}
	~BitMap() {
		delete[] id_offset;
		delete[] notzero_id;
		delete[] bitcode;
	}
};

class Graph {
public:
	std::vector<Vertex> vertices;
	unsigned vertex_num;
	unsigned edge_num;
	unsigned vertex_num_after_trim;
	unsigned edge_num_after_trim;
	CSR* csr;

	Graph() {
		vertex_num = 0;
		edge_num = 0;
		csr = NULL;
		vertex_num_after_trim = 0;
		edge_num_after_trim = 0;
	}

	void addVertex(unsigned lb);
	void addBipartiteEdge(unsigned lb1, unsigned lb2);
	void addEdge(unsigned lb1, unsigned lb2);
	void printGraph();
	void transformToCSR(CSR& _csr);
	void transformToBitMap(BitMap& _bitmap);
};

void
Graph::addVertex(unsigned lb) {
	vertices.push_back(Vertex(lb));
	vertex_num++;
}

void
Graph::addEdge(unsigned lb1, unsigned lb2) {
	vertices[lb1 - 1].neighbor.push_back(lb2);
	vertices[lb2 - 1].neighbor.push_back(lb1);
	edge_num++;
}

void
Graph::addBipartiteEdge(unsigned lb1, unsigned lb2) {
	vertices[lb1 - 1].neighbor.push_back(lb2);
	edge_num++;
}

void
Graph::printGraph() {
	std::cout << "Number of vertices: " << this->vertex_num << std::endl;
}

void
Graph::transformToCSR(CSR& _csr) {
	unsigned offset_size = this->vertex_num + 1;
	_csr.row_offset = new unsigned[offset_size];
	_csr.row_offset[0] = 0;
	unsigned sum = 0;
	for (int i = 1; i < offset_size; i++) {
		sum += this->vertices[i - 1].neighbor.size();
		_csr.row_offset[i] = sum;
	}
	//sum += this->vertices[offset_size - 1].neighbor.size();

	_csr.column_index = new unsigned[sum];
	unsigned k = 0;
	for (int i = 0; i < offset_size - 1; i++) {
		for (int j = 0; j < this->vertices[i].neighbor.size(); j++) {
			_csr.column_index[k] = this->vertices[i].neighbor[j];
			k++;
		}
	}
	this->csr = &_csr;
}

void
Graph::transformToBitMap(BitMap& _bitmap) {
	unsigned offset_size = this->vertex_num + 1;
	_bitmap.id_offset = new unsigned[offset_size];
	_bitmap.id_offset[0] = 0;
	unsigned sum = 0;
	for (int i = 0; i < offset_size - 1; i++) {
		int length = 32;
		if (this->vertices[i].neighbor.size() == 0) {
			_bitmap.id_offset[i + 1] = sum;
			continue;
		}
		while (length <= this->vertices[i].neighbor[0] - 1) length += 32;
		for (int j = 0; j < this->vertices[i].neighbor.size(); j++) {
			if (this->vertices[i].neighbor[j] - 1 >= length) {
				sum++;
				do {
					length += 32;
				} while (length <= this->vertices[i].neighbor[j] - 1);
			}
		}
		sum++;
		_bitmap.id_offset[i + 1] = sum;
	}
	_bitmap.notzero_id = new unsigned[sum]();
	_bitmap.bitcode = new unsigned[sum]();
	int count = 0;
	for (int i = 0; i < offset_size - 1; i++) {
		int last = -1;
		for (int j = 0; j < this->vertices[i].neighbor.size(); j++) {
			int offset = (this->vertices[i].neighbor[j] - 1) >> 5;
			int id = this->vertices[i].neighbor[j] - 1 - offset * 32;
			if (last != offset) {
				last = offset;
				_bitmap.notzero_id[count++] = offset;
			}
			_bitmap.bitcode[count - 1] |= (1 << id);
		}
		//printf("%d\n", count);
	}
}

void readFile(Graph& graph, bool flag, char* path) {
	FILE* fp = NULL;
	fp = fopen(path, "r");
	int L_vertex_num, R_vertex_num, edge_num;
	fscanf(fp, "%d %d %d\n", &L_vertex_num, &R_vertex_num, &edge_num);
	// if flag is true, input the left layer
	if (flag == true) {
		for (unsigned i = 1; i <= L_vertex_num; i++) {
			graph.addVertex(i);
		}
		for (int j = 0; j < edge_num; j++) {
			unsigned in, out;
			fscanf(fp, "%u %u\n", &in, &out);
			graph.addBipartiteEdge(in, out);
		}
	}
	else {
		for (unsigned i = 1; i <= R_vertex_num; i++) {
			graph.addVertex(i);
		}
		for (int j = 0; j < edge_num; j++) {
			unsigned in, out;
			fscanf(fp, "%u %u\n", &in, &out);
			graph.addBipartiteEdge(out, in);
		}
	}
	fclose(fp);
}

void Collect2Hop(Graph& L, Graph& R, Graph& H, int q) {
	//for L
	unsigned L_num = L.vertex_num;
	for (unsigned m = 1; m <= L_num; m++) {
		H.addVertex(m);
	}
	for (unsigned i = 0; i < L_num; i++) {
		// if(i % 1000 == 0) printf("Collect %d\n",i);
		if (L.vertices[i].label != 0) {
			unsigned* list2hop = new unsigned[L_num]();
			int i_neighbor_num = L.vertices[i].neighbor.size();
			for (int j = 0; j < i_neighbor_num; j++) {
				unsigned j_vertex = L.vertices[i].neighbor[j];
				int j_neighbor_num = R.vertices[j_vertex - 1].neighbor.size();
				for (int k = 0; k < j_neighbor_num; k++) {
					list2hop[R.vertices[j_vertex - 1].neighbor[k] - 1]++;
				}
			}
			for (unsigned l = i + 1; l < L_num; l++) {
				if (list2hop[l] >= q) {
					H.addEdge(i + 1, l + 1);
				}
			}
			delete[] list2hop;
		}
	}
}

int BinarySearch(std::vector<unsigned>& nums, int size, unsigned t) {
	int left = 0, right = size - 1;
	while (left <= right) {
		int mid = left + ((right - left) / 2);
		if (nums[mid] > t) {
			right = mid - 1;
		}
		else if (nums[mid] < t) {
			left = mid + 1;
		}
		else {
			return mid;
		}
	}
	return -1;
}

bool cmp(std::pair<unsigned, unsigned> a, std::pair<unsigned, unsigned> b) {
	return a.second < b.second;
}

void edgeDirectingByDegree(Graph& H) {
	std::vector<std::pair<unsigned, unsigned>> lb_degree;
	unsigned vertex_num = H.vertex_num;
	//int count = 0;
	for (unsigned i = 0; i < vertex_num; i++) {
		lb_degree.push_back(std::pair<unsigned, unsigned>(i + 1, H.vertices[i].neighbor.size()));
	}
	sort(lb_degree.begin(), lb_degree.end(), cmp);
	/*for (int j = vertex_num - 1; j >= 0; j--) {
		for (int k = j - 1; k >= 0; k--) {
			std::vector<unsigned>::iterator find_val = find(H.vertices[lb_degree[j].first - 1].neighbor.begin(), H.vertices[lb_degree[j].first - 1].neighbor.end(), lb_degree[k].first);
			if (find_val != H.vertices[lb_degree[j].first - 1].neighbor.end()) {
				H.vertices[lb_degree[j].first - 1].neighbor.erase(find_val);
				count1++;
			}
		}
	}
	std::cout << "Deleted:" << count1 << std::endl;*/
	for (int j = 0; j < vertex_num; j++) {
		for (auto val : H.vertices[lb_degree[j].first - 1].neighbor) {
			// std::vector<unsigned>::iterator find_val = find(H.vertices[val - 1].neighbor.begin(), H.vertices[val - 1].neighbor.end(), lb_degree[j].first);
			// if (find_val != H.vertices[val - 1].neighbor.end()) {
			// 	H.vertices[val - 1].neighbor.erase(find_val);
			// 	//count++;
			// }
			int res = BinarySearch(H.vertices[val - 1].neighbor, H.vertices[val - 1].neighbor.size(), lb_degree[j].first);
			if (res != -1){
				std::vector<unsigned>::iterator find_val = H.vertices[val - 1].neighbor.begin() + res;
				H.vertices[val - 1].neighbor.erase(find_val);
			}
		}
	}
	//std::cout << "Deleted:" << count << std::endl;
}

void TrimGraphByCoreNew(Graph& L, Graph& R, int p, int q) {
	int count = 0, old_count = -1;
	unsigned right_removed_edge = 0, left_removed_edge = 0;
	std::vector<unsigned> left_removed, right_removed;
	//std::vector<Vertex>tmp_left, tmp_right;
	while (count != old_count) {
		old_count = count;
		right_removed_edge = 0, left_removed_edge = 0;
		left_removed.clear();
		right_removed.clear();
		//Process left layer
		for (unsigned i = 0; i < L.vertex_num; i++) {
			if (L.vertices[i].label != 0 && L.vertices[i].neighbor.size() < q) {
				left_removed.push_back(i + 1);
				L.vertices[i].label = 0;
				count++;
			}
		}

		L.vertex_num_after_trim = L.vertex_num_after_trim == 0 ? L.vertex_num - left_removed.size() : L.vertex_num_after_trim - left_removed.size();

		//for (int j = 0; j < R.vertex_num; j++) {
		for (int i = 0; i < left_removed.size(); i++){
			unsigned re_vertex = left_removed[i];
			for (int j = 0; j < L.vertices[re_vertex - 1].neighbor.size(); j++) {
				unsigned vertex = L.vertices[re_vertex - 1].neighbor[j];
				int it = BinarySearch(R.vertices[vertex - 1].neighbor, R.vertices[vertex - 1].neighbor.size(), re_vertex);
				if (it != -1) {
					R.vertices[vertex - 1].neighbor.erase(R.vertices[vertex - 1].neighbor.begin() + it);
					left_removed_edge++;
				}
			}
			L.vertices[re_vertex - 1].neighbor.clear();
		}
		//printf("Now Left: %d\n", L.vertex_num_after_trim);

		L.edge_num_after_trim = L.edge_num_after_trim == 0 ? L.edge_num - left_removed_edge : L.edge_num_after_trim - left_removed_edge;
		R.edge_num_after_trim = R.edge_num_after_trim == 0 ? R.edge_num - left_removed_edge : R.edge_num_after_trim - left_removed_edge;
		//Right layer
		for (unsigned i = 0; i < R.vertex_num; i++) {
			if (R.vertices[i].label != 0 && R.vertices[i].neighbor.size() < p) {
				right_removed.push_back(i + 1);
				R.vertices[i].label = 0;
				count++;
			}
		}

		R.vertex_num_after_trim = R.vertex_num_after_trim == 0 ? R.vertex_num - right_removed.size() : R.vertex_num_after_trim - right_removed.size();

		for (int i = 0; i < right_removed.size(); i++) {
			unsigned re_vertex = right_removed[i];
			for (int j = 0; j < R.vertices[re_vertex - 1].neighbor.size(); j++) {
				unsigned vertex = R.vertices[re_vertex - 1].neighbor[j];
				//auto it = find(L.vertices[j].neighbor.begin(), L.vertices[j].neighbor.end(), right_removed[i]);
				int it = BinarySearch(L.vertices[vertex - 1].neighbor, L.vertices[vertex - 1].neighbor.size(), re_vertex);
				if (it != -1) {
					L.vertices[vertex - 1].neighbor.erase(L.vertices[vertex - 1].neighbor.begin() + it);
					right_removed_edge++;
				}
			}
			R.vertices[re_vertex - 1].neighbor.clear();
		}
		//printf("Now Right: %d\n", R.vertex_num_after_trim);

		R.edge_num_after_trim = R.edge_num_after_trim == 0 ? R.edge_num - right_removed_edge : R.edge_num_after_trim - right_removed_edge;
		L.edge_num_after_trim = L.edge_num_after_trim == 0 ? L.edge_num - right_removed_edge : L.edge_num_after_trim - right_removed_edge;
	}
}

void reformatGraph(Graph& L, Graph& R) {
	std::map<unsigned, unsigned> dic_L, dic_R;
	unsigned count_L = 1, count_R = 1;
	for (unsigned i = 0; i < L.vertex_num; i++) {
		if (L.vertices[i].label != 0) {
			dic_L[i + 1] = count_L;
			count_L++;
		}
	}
	for (unsigned i = 0; i < R.vertex_num; i++) {
		if (R.vertices[i].label != 0) {
			dic_R[i + 1] = count_R;
			count_R++;
		}
	}
	// printf("%d %d\n", count_L - 1, count_R - 1);
	count_L = 0, count_R = 0;
	for (unsigned i = 0; i < L.vertex_num; i++) {
		for (unsigned j = 0; j < L.vertices[i].neighbor.size(); j++) {
			L.vertices[i].neighbor[j] = dic_R[L.vertices[i].neighbor[j]];
			//count_L++;
		}
	}
	for (unsigned i = 0; i < R.vertex_num; i++) {
		for (unsigned j = 0; j < R.vertices[i].neighbor.size(); j++) {
			R.vertices[i].neighbor[j] = dic_L[R.vertices[i].neighbor[j]];
			//count_L++;
		}
	}
	for (unsigned i = 0; i < L.vertex_num; i++) {
		if (L.vertices[i].label != 0) {
			L.vertices[count_L] = L.vertices[i];
			count_L++;
		}
	}
	for (unsigned i = 0; i < R.vertex_num; i++) {
		if (R.vertices[i].label != 0) {
			R.vertices[count_R] = R.vertices[i];
			count_R++;
		}
	}
	// printf("%d %d\n", count_L, count_R);
	L.vertex_num = L.vertex_num_after_trim;
	R.vertex_num = R.vertex_num_after_trim;
	L.edge_num = L.edge_num_after_trim;
	R.edge_num = R.edge_num_after_trim;
	dic_L.clear();
	dic_R.clear();
}

__device__ unsigned stack[BLOCKNUM][MAXSTACKSIZE] = { 0 };
__device__ unsigned stack_bitmap[BLOCKNUM][MAXSTACKSIZE][2] = { 0 }; // 0 is offset, 1 is bit

__device__ unsigned subH[BLOCKNUM][MAXSTACKSIZE][MAXHSIZE] = { 0 };
__device__ unsigned S[BLOCKNUM][MAXSTACKSIZE][MAXSSIZE] = { 0 };

__device__ unsigned subHBits[BLOCKNUM][MAXSTACKSIZE][MAXHSIZE] = { 0 };
__device__ unsigned SBits[BLOCKNUM][MAXSTACKSIZE][MAXSSIZE] = { 0 };

__device__ unsigned offset_H[BLOCKNUM][MAXSTACKSIZE][MAXHSIZE] = { 0 };
__device__ unsigned offset_L[BLOCKNUM][MAXSTACKSIZE][MAXSSIZE] = { 0 };

__device__ unsigned batch_info[BLOCKNUM][MAXSTACKSIZE][3] = { 0 }; // 0 is batch index, 1 is batch size, 2 is next_k

__device__ unsigned glockArray[BLOCKNUM] = { 0 };
__device__ unsigned GCL[BLOCKNUM] = { 0 };

__device__  unsigned long long OrderMulDev(unsigned m, unsigned n) {
	// if (n == 0 || n == m) {
	// 	return 1;
	// }
	// return OrderMulDev(m - 1, n) + OrderMulDev(m - 1, n - 1);
	unsigned long long ans = 1;
    for(unsigned i = 1; i <= n; i++){
        ans *= m--;
		ans /= i; 
    }
    return ans;
}

__device__ unsigned CalNumOfOne(unsigned* BitsArray, unsigned num) {
	unsigned sum = 0;
	for (unsigned i = 0; i < num; i++) {
		unsigned xx = BitsArray[i];
		xx = xx - ((xx >> 1) & 0x55555555);
		xx = (xx & 0x33333333) + ((xx >> 2) & 0x33333333);
		xx = (xx + (xx >> 4)) & 0x0f0f0f0f;
		xx = xx + (xx >> 8);
		sum += (xx + (xx >> 16)) & 0xff;
		// unsigned ans = 0;
        // while (n != 0) {
        //     ans += (n & 1);
        //     n >>= 1;
        // }
        // sum += ans;
	}
	return sum;
}

__device__ void IntersectionDevBitsOld2(unsigned* B, unsigned B_size, unsigned A[], unsigned Res_size, int Res[], unsigned* BBits, unsigned* ABits, unsigned Res_Bits[]) {
	int tid_t = threadIdx.x < 32 ? threadIdx.x : threadIdx.x - 32;
	//printf("%d ", tid);
	//unsigned long long start, end, start2, end2;
	while (tid_t < Res_size) {
		//start = clock64() / CLOCKF;
		int val = A[tid_t];
		//binary search
		int ret = 0, tmp = B_size;
		while (tmp > 1) {
			int halfsize = tmp / 2;
			int cand = B[ret + halfsize];
			ret += (cand < val) ? halfsize : 0;
			tmp -= halfsize;
		}
		//end = clock64() / CLOCKF;
		//duration += (end - start);
		ret += (B[ret] < val);
		//start = clock64() / CLOCKF;
		unsigned tmp_bits = BBits[ret] & ABits[tid_t];
		Res[tid_t] = (ret <= (B_size - 1) ? (B[ret] == val) ? (tmp_bits != 0) ? val : -1 : -1 : -1);

		/*end = clock64() / CLOCKF;
		duration += (end - start);*/

		Res_Bits[tid_t] = tmp_bits;
		tid_t += (blockDim.x / 2);
	}
}

__device__ void IntersectionDevBitsOld3(unsigned* B, unsigned B_size, int Res[], unsigned Res_size, unsigned* BBits, unsigned Res_Bits[]) {
	int tid_t = threadIdx.x < 32 ? threadIdx.x : threadIdx.x - 32;
	//printf("%d ", tid);
	//unsigned long long start, end, start2, end2;
	while (tid_t < Res_size) {
		//start = clock64() / CLOCKF;
		int val = Res[tid_t];
		//binary search
		int ret = 0, tmp = B_size;
		while (tmp > 1) {
			int halfsize = tmp / 2;
			int cand = B[ret + halfsize];
			ret += (cand < val) ? halfsize : 0;
			tmp -= halfsize;
		}
		//end = clock64() / CLOCKF;
		//duration += (end - start);
		ret += (B[ret] < val);
		//start = clock64() / CLOCKF;
		unsigned tmp_bits = BBits[ret] & Res_Bits[tid_t];
		Res[tid_t] = (ret <= (B_size - 1) ? (B[ret] == val) ? (tmp_bits != 0) ? val : -1 : -1 : -1);

		/*end = clock64() / CLOCKF;
		duration += (end - start);*/

		Res_Bits[tid_t] = tmp_bits;
		tid_t += (blockDim.x / 2);
	}
}

__device__ void IntersectionDevBits6(unsigned* A, unsigned* begin, unsigned* end, int* Res, int Res_num, unsigned size, unsigned* ABits, unsigned* Res_Bits) {
	int tid_t = threadIdx.x < 32 ? threadIdx.x : threadIdx.x - 32;
	//printf("%d ", tid);
	//unsigned long long start, end, start2, end2;
	while (tid_t < Res_num * size) {
		//start = clock64() / CLOCKF;
		int idx = tid_t / Res_num;
		int A_size = end[idx] - begin[idx];
		int val = Res[tid_t];
		//binary search
		int ret = 0, tmp = A_size;
		while (tmp > 1) {
			int halfsize = tmp / 2;
			int cand = A[begin[idx] + ret + halfsize];
			ret += (cand < val) ? halfsize : 0;
			tmp -= halfsize;
		}
		//end = clock64() / CLOCKF;
		//duration += (end - start);
		ret += (A[begin[idx] + ret] < val);
		//start = clock64() / CLOCKF;
		unsigned tmp_bits = ABits[begin[idx] + ret] & Res_Bits[tid_t];
		Res[tid_t] = (ret <= (A_size - 1) ? (A[begin[idx] + ret] == val) ? (tmp_bits != 0) ? val : -1 : -1 : -1);

		/*end = clock64() / CLOCKF;
		duration += (end - start);*/

		Res_Bits[tid_t] = tmp_bits;
		tid_t += (blockDim.x / 2);
	}
}

__global__ void findCliqueGPUNew7(unsigned* id_offset_dev_L, unsigned* notzero_id_dev_L, unsigned* bits_dev_L, unsigned* id_offset_dev_H, unsigned* notzero_id_dev_H, unsigned* bits_dev_H, unsigned long long* count, unsigned* p, unsigned* q, unsigned* Hsize, unsigned* non_vertex, unsigned* non_vertex_father) {
	__shared__ int top, level;
	__shared__ unsigned next_k, tid;
	__shared__ unsigned Res_SBits[2000], Res_HBits[1000], begin_L[100], end_L[100], begin_H[100], end_H[100], Num_L[8][100], Num_H[8][100], Num_L_Bits[8][100], Num_H_Bits[8][100];
	__shared__ int Res_S[2000], Res_H[1000];
	tid = blockIdx.x;
	// unsigned long long start, end, start1, end1, start2, end2, duration = 0, duration1 = 0, duration2 = 0;
	__syncthreads();
	while (tid < *Hsize) {
		__syncthreads();
		if (threadIdx.x == 0) {
			top = 0;
			level = 0;
			next_k = 0;
			stack[blockIdx.x][top] = 0;
			// printf("%d:%d\n", blockIdx.x, tid);
		}
		__syncthreads();
		if (threadIdx.x < 32) {
			unsigned num_S = id_offset_dev_L[non_vertex[tid] + 1] - id_offset_dev_L[non_vertex[tid]], num_L = id_offset_dev_L[non_vertex_father[tid] + 1] - id_offset_dev_L[non_vertex_father[tid]];
			if (num_S < num_L) {
				IntersectionDevBitsOld2(&notzero_id_dev_L[id_offset_dev_L[non_vertex_father[tid]]], num_L, &notzero_id_dev_L[id_offset_dev_L[non_vertex[tid]]], num_S, Res_S, &bits_dev_L[id_offset_dev_L[non_vertex_father[tid]]], &bits_dev_L[id_offset_dev_L[non_vertex[tid]]], Res_SBits);
			}
			else {
				IntersectionDevBitsOld2(&notzero_id_dev_L[id_offset_dev_L[non_vertex[tid]]], num_S, &notzero_id_dev_L[id_offset_dev_L[non_vertex_father[tid]]], num_L, Res_S, &bits_dev_L[id_offset_dev_L[non_vertex[tid]]], &bits_dev_L[id_offset_dev_L[non_vertex_father[tid]]], Res_SBits);
			}
			if (threadIdx.x == 0) {
				Num_L[level][0] = 0;
				unsigned tmp_interval = num_S > num_L ? num_L : num_S;
				for (int k = 0; k < tmp_interval; k++) {
					if (Res_S[k] != -1) {
						S[blockIdx.x][level][++Num_L[level][0]] = Res_S[k];
						SBits[blockIdx.x][level][Num_L[level][0]] = Res_SBits[k];
					}
				}
				Num_L_Bits[level][0] = CalNumOfOne(&SBits[blockIdx.x][level][1], Num_L[level][0]);
				S[blockIdx.x][level][0] = 1;
			}
		}
		else {
			unsigned num_subH = id_offset_dev_H[non_vertex[tid] + 1] - id_offset_dev_H[non_vertex[tid]], num_H = id_offset_dev_H[non_vertex_father[tid] + 1] - id_offset_dev_H[non_vertex_father[tid]];
			if (num_subH < num_H) {
				IntersectionDevBitsOld2(&notzero_id_dev_H[id_offset_dev_H[non_vertex_father[tid]]], num_H, &notzero_id_dev_H[id_offset_dev_H[non_vertex[tid]]], num_subH, Res_H, &bits_dev_H[id_offset_dev_H[non_vertex_father[tid]]], &bits_dev_H[id_offset_dev_H[non_vertex[tid]]], Res_HBits);
			}
			else {
				IntersectionDevBitsOld2(&notzero_id_dev_H[id_offset_dev_H[non_vertex[tid]]], num_subH, &notzero_id_dev_H[id_offset_dev_H[non_vertex_father[tid]]], num_H, Res_H, &bits_dev_H[id_offset_dev_H[non_vertex[tid]]], &bits_dev_H[id_offset_dev_H[non_vertex_father[tid]]], Res_HBits);
			}
			if (threadIdx.x == 32) {
				Num_H[level][0] = 0;
				unsigned tmp_interval = num_subH > num_H ? num_H : num_subH;
				for (int k = 0; k < tmp_interval; k++) {
					if (Res_H[k] != -1) {
						subH[blockIdx.x][level][++Num_H[level][0]] = Res_H[k];
						subHBits[blockIdx.x][level][Num_H[level][0]] = Res_HBits[k];
					}
				}
				Num_H_Bits[level][0] = CalNumOfOne(&subHBits[blockIdx.x][level][1], Num_H[level][0]);
				subH[blockIdx.x][level][0] = 1;
			}
		}
		__syncthreads();
		if (Num_L_Bits[level][0] >= *q && Num_H_Bits[level][0] >= *p - level - 2) {
			__syncthreads();
			if (threadIdx.x == 0) {
				level++;
				top++;
			}
			__syncthreads();
		}
		__syncthreads();
		while (top != 0) {
			__syncthreads();
			int size = Num_H_Bits[level - 1][next_k];
			// __syncthreads();
			if (threadIdx.x == 0) {
				SBits[blockIdx.x][level][0] = size;
				subHBits[blockIdx.x][level][0] = size;
				S[blockIdx.x][level][0] = Num_H[level - 1][next_k];
				subH[blockIdx.x][level][0] = Num_H[level - 1][next_k];
				/*if (size > 100) {
					printf("AA\n");
				}*/
				//printf("%d-%d-%d-%d\n", blockIdx.x, tid, level, next_k);
				/*if (tid == 8 && level == 7 && next_k == 4) {
					printf("A\n");
				}*/
				// printf("%d\n",next_k);
				// printf("A\n");
				int count_tmp = 0;
				for (int j = 0; j < Num_H[level - 1][next_k]; j++) {
					unsigned tmp_offset_bitmap = subH[blockIdx.x][level - 1][offset_H[blockIdx.x][level - 1][next_k] + j + 1];
					unsigned bit_num = subHBits[blockIdx.x][level - 1][offset_H[blockIdx.x][level - 1][next_k] + j + 1];
					for (int k = 0; k < 32; k++) {
						if ((bit_num & (1 << k)) != 0) {
							unsigned vertex_tmp = k + 32 * tmp_offset_bitmap;
							end_L[count_tmp] = id_offset_dev_L[vertex_tmp + 1];
							begin_L[count_tmp] = id_offset_dev_L[vertex_tmp];

							end_H[count_tmp] = id_offset_dev_H[vertex_tmp + 1];
							begin_H[count_tmp] = id_offset_dev_H[vertex_tmp];
							count_tmp++;
						}
					}
				}
			}
			__syncthreads();
			if (threadIdx.x < 32) {
				unsigned inter_offset = stack[blockIdx.x][level - 1];
				inter_offset = (inter_offset == 0 ? 1 : inter_offset);
				unsigned inter_idx = offset_L[blockIdx.x][level - 1][inter_offset - 1];
				unsigned* begin_S = &S[blockIdx.x][level - 1][inter_idx + 1];
				unsigned* begin_SBits = &SBits[blockIdx.x][level - 1][inter_idx + 1];
				unsigned num_S = Num_L[level - 1][inter_offset - 1];
				// unsigned num_SBits = Num_L_Bits[level - 1][inter_offset - 1];
				for (int i = threadIdx.x; i < size; i += (THREADNUM / 2)) {
					offset_L[blockIdx.x][level][i] = i * num_S;
				}
				if (num_S < 32) {
					// 将S扩至size倍
					for (int i = 0; i < size; i++) {
						if (threadIdx.x < num_S) {
							Res_S[threadIdx.x + i * num_S] = begin_S[threadIdx.x];
							Res_SBits[threadIdx.x + i * num_S] = begin_SBits[threadIdx.x];
						}
					}
					IntersectionDevBits6(notzero_id_dev_L, begin_L, end_L, Res_S, num_S, size, bits_dev_L, Res_SBits);
					for (int i = threadIdx.x; i < size; i += (THREADNUM / 2)) {
						int res_offset = i * num_S;
						Num_L[level][i] = 0;
						for (int k = 0; k < num_S; k++) {
							if (Res_S[k + res_offset] != -1) {
								S[blockIdx.x][level][res_offset + (++Num_L[level][i])] = Res_S[k + res_offset];
								SBits[blockIdx.x][level][res_offset + (Num_L[level][i])] = Res_SBits[k + res_offset];
							}
						}
						Num_L_Bits[level][i] = CalNumOfOne(&SBits[blockIdx.x][level][res_offset + 1], Num_L[level][i]);
					}
					//if (threadIdx.x == 0) if (size * num_S > 2000) printf("AAAA: %d,%d\n", num_S, size);
				}
				else {
					for (int i = 0; i < size; i++) {
						IntersectionDevBitsOld2(&notzero_id_dev_L[begin_L[i]], end_L[i] - begin_L[i], begin_S, num_S, Res_S, &bits_dev_L[begin_L[i]], begin_SBits, Res_SBits);
						if (threadIdx.x == 0) {
							int res_offset = i * num_S;
							Num_L[level][i] = 0;
							for (int k = 0; k < num_S; k++) {
								if (Res_S[k] != -1) {
									S[blockIdx.x][level][res_offset + (++Num_L[level][i])] = Res_S[k];
									SBits[blockIdx.x][level][res_offset + (Num_L[level][i])] = Res_SBits[k];
								}
							}
							Num_L_Bits[level][i] = CalNumOfOne(&SBits[blockIdx.x][level][res_offset + 1], Num_L[level][i]);
						}
					}
				}
			}
			else {
				unsigned inter_offset = stack[blockIdx.x][level - 1];
				inter_offset = (inter_offset == 0 ? 1 : inter_offset);
				unsigned inter_idx = offset_H[blockIdx.x][level - 1][inter_offset - 1];
				unsigned* begin_subH = &subH[blockIdx.x][level - 1][inter_idx + 1];
				unsigned* begin_subHBits = &subHBits[blockIdx.x][level - 1][inter_idx + 1];
				unsigned num_subH = Num_H[level - 1][inter_offset - 1];
				// unsigned num_subHBits = Num_H_Bits[level - 1][inter_offset - 1];
				for (int i = threadIdx.x - 32; i < size; i += (THREADNUM / 2)) {
					offset_H[blockIdx.x][level][i] = i * num_subH;
				}
				if (num_subH < 32) {
					for (int i = 0; i < size; i++) {
						if (threadIdx.x - 32 < num_subH) {
							Res_H[threadIdx.x - 32 + i * num_subH] = begin_subH[threadIdx.x - 32];
							Res_HBits[threadIdx.x - 32 + i * num_subH] = begin_subHBits[threadIdx.x - 32];
						}
					}
					//IntersectionDev6(col_dev_H, begin_H, end_H, Res_H, num_subH, num_subH);
					IntersectionDevBits6(notzero_id_dev_H, begin_H, end_H, Res_H, num_subH, size, bits_dev_H, Res_HBits);
					for (int i = threadIdx.x - 32; i < size; i += (THREADNUM / 2)) {
						int res_offset = i * num_subH;
						Num_H[level][i] = 0;
						for (int k = 0; k < num_subH; k++) {
							if (Res_H[k + res_offset] != -1) {
								subH[blockIdx.x][level][res_offset + (++Num_H[level][i])] = Res_H[k + res_offset];
								subHBits[blockIdx.x][level][res_offset + (Num_H[level][i])] = Res_HBits[k + res_offset];
							}
						}
						Num_H_Bits[level][i] = CalNumOfOne(&subHBits[blockIdx.x][level][res_offset + 1], Num_H[level][i]);
					}
					//if (threadIdx.x == 32) if (size * num_subH > 2000) printf("BBBB: %d,%d\n", num_subH, size);
				}
				else {
					for (int i = 0; i < size; i++) {
						//IntersectionDevOld2(&col_dev_H[begin_H[i]], end_H[i] - begin_H[i], begin_subH, num_subH, Res_H);
						IntersectionDevBitsOld2(&notzero_id_dev_H[begin_H[i]], end_H[i] - begin_H[i], begin_H, num_subH, Res_H, &bits_dev_H[begin_H[i]], begin_subHBits, Res_HBits);
						if (threadIdx.x == 32) {
							int res_offset = i * num_subH;
							Num_H[level][i] = 0;
							for (int k = 0; k < num_subH; k++) {
								if (Res_H[k] != -1) {
									subH[blockIdx.x][level][res_offset + (++Num_H[level][i])] = Res_H[k];
									subHBits[blockIdx.x][level][res_offset + (Num_H[level][i])] = Res_HBits[k];
								}
							}
							Num_H_Bits[level][i] = CalNumOfOne(&subHBits[blockIdx.x][level][res_offset + 1], Num_H[level][i]);
						}
					}
				}
			}
			__syncthreads();
			if (level == *p - 2) {
				for (int i = threadIdx.x; i < size; i += blockDim.x) {
					unsigned m = Num_L_Bits[level][i];
					if (m >= *q) {
						atomicAdd(count, OrderMulDev(m, *q));
					}
				}
				__syncthreads();
				if (threadIdx.x == 0) {
					//stack[blockIdx.x][top] = 0;
					stack[blockIdx.x][level] = 0;
					top--;
					level--;
				}
			}
			__syncthreads();
			if (threadIdx.x == 0) {
				while (top != 0) {
					int flag = 0, loop_size = subHBits[blockIdx.x][level][0];
					for (int k = stack[blockIdx.x][level]; k < loop_size; k++) {
						if (Num_L_Bits[level][k] >= *q && Num_H_Bits[level][k] >= *p - level - 2) {
							stack[blockIdx.x][level] = k + 1;
							next_k = k;
							flag = 1;
							top++;
							level++;
							break;
						}
					}
					if (flag == 0) {
						stack[blockIdx.x][level] = 0;
						level--;
						top--;
					}
					else {
						break;
					}
				}
			}
			__syncthreads();
		}
		if (threadIdx.x == 0) {
			tid += gridDim.x;
		}
		__syncthreads();
	}
	/*if (threadIdx.x == 0) {
		printf("%d end\n", blockIdx.x);
	}*/
}

__global__ void findCliqueGPUNewBatch(unsigned* id_offset_dev_L, unsigned* notzero_id_dev_L, unsigned* bits_dev_L, unsigned* id_offset_dev_H, unsigned* notzero_id_dev_H, unsigned* bits_dev_H, unsigned long long* count, unsigned* p, unsigned* q, unsigned* Hsize, unsigned* non_vertex, unsigned* non_vertex_father) {
	__shared__ int top, level;
	__shared__ unsigned next_k, tid, batch_size;
	__shared__ unsigned Res_SBits[MAXSBATCHSIZE], Res_HBits[MAXHBATCHSIZE], begin_L[MAXBATCHLEVELSIZE], end_L[MAXBATCHLEVELSIZE], begin_H[MAXBATCHLEVELSIZE], end_H[MAXBATCHLEVELSIZE], Num_L[MAXSTACKSIZE][MAXBATCHLEVELSIZE], Num_H[MAXSTACKSIZE][MAXBATCHLEVELSIZE], Num_L_Bits[MAXSTACKSIZE][MAXBATCHLEVELSIZE], Num_H_Bits[MAXSTACKSIZE][MAXBATCHLEVELSIZE];
	__shared__ int Res_S[MAXSBATCHSIZE], Res_H[MAXHBATCHSIZE];
	tid = blockIdx.x;
	// unsigned long long start, end, start1, end1, start2, end2, duration = 0, duration1 = 0, duration2 = 0;
	__syncthreads();
	while (tid < *Hsize) {
		__syncthreads();
		if (threadIdx.x == 0) {
			top = 0;
			level = 0;
			next_k = 0;
			stack[blockIdx.x][top] = 0;
			// printf("%d:%d\n", blockIdx.x, tid);
		}
		__syncthreads();
		if (threadIdx.x < 32) {
			unsigned num_S = id_offset_dev_L[non_vertex[tid] + 1] - id_offset_dev_L[non_vertex[tid]], num_L = id_offset_dev_L[non_vertex_father[tid] + 1] - id_offset_dev_L[non_vertex_father[tid]];
			if (num_S < num_L) {
				unsigned batch_begin_S = num_S / MAXSBATCHSIZE;
				unsigned last_batch = num_S - batch_begin_S * MAXSBATCHSIZE;
				if (threadIdx.x == 0) {
					Num_L[level][0] = 0;
					// Num_L_Bits[level][0] = 0;
				}
				for (int i = 0; i < batch_begin_S; i++) {
					// 填充Res_S
					for (int j = threadIdx.x; j < MAXSBATCHSIZE; j += (THREADNUM / 2)) {
						// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + i * MAXSBATCHSIZE];
						Res_S[j] = notzero_id_dev_L[id_offset_dev_L[non_vertex[tid]] + j + i * MAXSBATCHSIZE];
						Res_SBits[j] = bits_dev_L[id_offset_dev_L[non_vertex[tid]] + j + i * MAXSBATCHSIZE];
					}
					// 计算该Batch的交集
					// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, MAXSBATCHSIZE);
					IntersectionDevBitsOld3(&notzero_id_dev_L[id_offset_dev_L[non_vertex_father[tid]]], num_L, Res_S, MAXSBATCHSIZE, &bits_dev_L[id_offset_dev_L[non_vertex_father[tid]]], Res_SBits);
					if (threadIdx.x == 0) {
						for (int k = 0; k < MAXSBATCHSIZE; k++) {
							if (Res_S[k] != -1) {
								S[blockIdx.x][level][++Num_L[level][0]] = Res_S[k];
								SBits[blockIdx.x][level][Num_L[level][0]] = Res_SBits[k];
							}
						}
					}
				}
				// 计算最后一个batch
				for (int j = threadIdx.x; j < last_batch; j += (THREADNUM / 2)) {
					// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + batch_begin_S * MAXSBATCHSIZE];
					Res_S[j] = notzero_id_dev_L[id_offset_dev_L[non_vertex[tid]] + j + batch_begin_S * MAXSBATCHSIZE];
					Res_SBits[j] = bits_dev_L[id_offset_dev_L[non_vertex[tid]] + j + batch_begin_S * MAXSBATCHSIZE];
				}
				// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, last_batch);
				IntersectionDevBitsOld3(&notzero_id_dev_L[id_offset_dev_L[non_vertex_father[tid]]], num_L, Res_S, last_batch, &bits_dev_L[id_offset_dev_L[non_vertex_father[tid]]], Res_SBits);
				if (threadIdx.x == 0) {
					for (int k = 0; k < last_batch; k++) {
						if (Res_S[k] != -1) {
							S[blockIdx.x][level][++Num_L[level][0]] = Res_S[k];
							SBits[blockIdx.x][level][Num_L[level][0]] = Res_SBits[k];
						}
					}
					Num_L_Bits[level][0] = CalNumOfOne(&SBits[blockIdx.x][level][1], Num_L[level][0]);
					// SBits[blockIdx.x][level][0] = Num_L_Bits[level][0];
				}
				// IntersectionDevBitsOld2(&notzero_id_dev_L[id_offset_dev_L[non_vertex_father[tid]]], num_L, &notzero_id_dev_L[id_offset_dev_L[non_vertex[tid]]], num_S, Res_S, &bits_dev_L[id_offset_dev_L[non_vertex_father[tid]]], &bits_dev_L[id_offset_dev_L[non_vertex[tid]]], Res_SBits);
			}
			else {
				// IntersectionDevBitsOld2(&notzero_id_dev_L[id_offset_dev_L[non_vertex[tid]]], num_S, &notzero_id_dev_L[id_offset_dev_L[non_vertex_father[tid]]], num_L, Res_S, &bits_dev_L[id_offset_dev_L[non_vertex[tid]]], &bits_dev_L[id_offset_dev_L[non_vertex_father[tid]]], Res_SBits);
				unsigned batch_begin_L = num_L / MAXSBATCHSIZE;
				unsigned last_batch = num_L - batch_begin_L * MAXSBATCHSIZE;
				if (threadIdx.x == 0) {
					Num_L[level][0] = 0;
				}
				for (int i = 0; i < batch_begin_L; i++) {
					// 填充Res_S
					for (int j = threadIdx.x; j < MAXSBATCHSIZE; j += (THREADNUM / 2)) {
						// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + i * MAXSBATCHSIZE];
						Res_S[j] = notzero_id_dev_L[id_offset_dev_L[non_vertex_father[tid]] + j + i * MAXSBATCHSIZE];
						Res_SBits[j] = bits_dev_L[id_offset_dev_L[non_vertex_father[tid]] + j + i * MAXSBATCHSIZE];
					}
					// 计算该Batch的交集
					// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, MAXSBATCHSIZE);
					IntersectionDevBitsOld3(&notzero_id_dev_L[id_offset_dev_L[non_vertex[tid]]], num_S, Res_S, MAXSBATCHSIZE, &bits_dev_L[id_offset_dev_L[non_vertex[tid]]], Res_SBits);
					if (threadIdx.x == 0) {
						for (int k = 0; k < MAXSBATCHSIZE; k++) {
							if (Res_S[k] != -1) {
								S[blockIdx.x][level][++Num_L[level][0]] = Res_S[k];
								SBits[blockIdx.x][level][Num_L[level][0]] = Res_SBits[k];
							}
						}
						// Num_L_Bits[level][0] += CalNumOfOne(&SBits[blockIdx.x][level][1], Num_L[level][0]);
					}
				}
				// 计算最后一个batch
				for (int j = threadIdx.x; j < last_batch; j += (THREADNUM / 2)) {
					// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + batch_begin_S * MAXSBATCHSIZE];
					Res_S[j] = notzero_id_dev_L[id_offset_dev_L[non_vertex_father[tid]] + j + batch_begin_L * MAXSBATCHSIZE];
					Res_SBits[j] = bits_dev_L[id_offset_dev_L[non_vertex_father[tid]] + j + batch_begin_L * MAXSBATCHSIZE];
				}
				// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, last_batch);
				IntersectionDevBitsOld3(&notzero_id_dev_L[id_offset_dev_L[non_vertex[tid]]], num_S, Res_S, last_batch, &bits_dev_L[id_offset_dev_L[non_vertex[tid]]], Res_SBits);
				if (threadIdx.x == 0) {
					for (int k = 0; k < last_batch; k++) {
						if (Res_S[k] != -1) {
							S[blockIdx.x][level][++Num_L[level][0]] = Res_S[k];
							SBits[blockIdx.x][level][Num_L[level][0]] = Res_SBits[k];
						}
					}
					Num_L_Bits[level][0] = CalNumOfOne(&SBits[blockIdx.x][level][1], Num_L[level][0]);
					// SBits[blockIdx.x][level][0] = Num_L_Bits[level][0];
				}
			}
		}
		else {
			unsigned num_subH = id_offset_dev_H[non_vertex[tid] + 1] - id_offset_dev_H[non_vertex[tid]], num_H = id_offset_dev_H[non_vertex_father[tid] + 1] - id_offset_dev_H[non_vertex_father[tid]];
			if (num_subH < num_H) {
				// IntersectionDevBitsOld2(&notzero_id_dev_H[id_offset_dev_H[non_vertex_father[tid]]], num_H, &notzero_id_dev_H[id_offset_dev_H[non_vertex[tid]]], num_subH, Res_H, &bits_dev_H[id_offset_dev_H[non_vertex_father[tid]]], &bits_dev_H[id_offset_dev_H[non_vertex[tid]]], Res_HBits);
				unsigned batch_begin_subH = num_subH / MAXHBATCHSIZE;
				unsigned last_batch = num_subH - batch_begin_subH * MAXHBATCHSIZE;
				if (threadIdx.x == 32) {
					Num_H[level][0] = 0;
					// Num_H_Bits[level][0] = 0;
				}
				for (int i = 0; i < batch_begin_subH; i++) {
					// 填充Res_S
					for (int j = threadIdx.x - 32; j < MAXHBATCHSIZE; j += (THREADNUM / 2)) {
						// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + i * MAXSBATCHSIZE];
						Res_H[j] = notzero_id_dev_H[id_offset_dev_H[non_vertex[tid]] + j + i * MAXHBATCHSIZE];
						Res_HBits[j] = bits_dev_H[id_offset_dev_H[non_vertex[tid]] + j + i * MAXHBATCHSIZE];
					}
					// 计算该Batch的交集
					// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, MAXSBATCHSIZE);
					IntersectionDevBitsOld3(&notzero_id_dev_H[id_offset_dev_H[non_vertex_father[tid]]], num_H, Res_H, MAXHBATCHSIZE, &bits_dev_H[id_offset_dev_H[non_vertex_father[tid]]], Res_HBits);
					if (threadIdx.x == 32) {
						for (int k = 0; k < MAXHBATCHSIZE; k++) {
							if (Res_H[k] != -1) {
								subH[blockIdx.x][level][++Num_H[level][0]] = Res_H[k];
								subHBits[blockIdx.x][level][Num_H[level][0]] = Res_HBits[k];
							}
						}
						// Num_H_Bits[level][0] += CalNumOfOne(&subHBits[blockIdx.x][level][1], Num_H[level][0]);
					}
				}
				// 计算最后一个batch
				for (int j = threadIdx.x - 32; j < last_batch; j += (THREADNUM / 2)) {
					// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + batch_begin_S * MAXSBATCHSIZE];
					Res_H[j] = notzero_id_dev_H[id_offset_dev_H[non_vertex[tid]] + j + batch_begin_subH * MAXHBATCHSIZE];
					Res_HBits[j] = bits_dev_H[id_offset_dev_H[non_vertex[tid]] + j + batch_begin_subH * MAXHBATCHSIZE];
				}
				// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, last_batch);
				IntersectionDevBitsOld3(&notzero_id_dev_H[id_offset_dev_H[non_vertex_father[tid]]], num_H, Res_H, last_batch, &bits_dev_H[id_offset_dev_H[non_vertex_father[tid]]], Res_HBits);
				if (threadIdx.x == 32) {
					for (int k = 0; k < last_batch; k++) {
						if (Res_H[k] != -1) {
							subH[blockIdx.x][level][++Num_H[level][0]] = Res_H[k];
							subHBits[blockIdx.x][level][Num_H[level][0]] = Res_HBits[k];
						}
					}
					Num_H_Bits[level][0] = CalNumOfOne(&subHBits[blockIdx.x][level][1], Num_H[level][0]);
					// subHBits[blockIdx.x][level][0] = Num_H_Bits[level][0];
				}
			}
			else {
				// IntersectionDevBitsOld2(&notzero_id_dev_H[id_offset_dev_H[non_vertex[tid]]], num_subH, &notzero_id_dev_H[id_offset_dev_H[non_vertex_father[tid]]], num_H, Res_H, &bits_dev_H[id_offset_dev_H[non_vertex[tid]]], &bits_dev_H[id_offset_dev_H[non_vertex_father[tid]]], Res_HBits);
				unsigned batch_begin_H = num_H / MAXHBATCHSIZE;
				unsigned last_batch = num_H - batch_begin_H * MAXHBATCHSIZE;
				if (threadIdx.x == 32) {
					Num_H[level][0] = 0;
				}
				for (int i = 0; i < batch_begin_H; i++) {
					// 填充Res_S
					for (int j = threadIdx.x - 32; j < MAXHBATCHSIZE; j += (THREADNUM / 2)) {
						Res_H[j] = notzero_id_dev_H[id_offset_dev_H[non_vertex_father[tid]] + j + i * MAXHBATCHSIZE];
						Res_HBits[j] = bits_dev_H[id_offset_dev_H[non_vertex_father[tid]] + j + i * MAXHBATCHSIZE];
					}
					// 计算该Batch的交集
					IntersectionDevBitsOld3(&notzero_id_dev_H[id_offset_dev_H[non_vertex[tid]]], num_subH, Res_H, MAXHBATCHSIZE, &bits_dev_H[id_offset_dev_H[non_vertex[tid]]], Res_HBits);
					if (threadIdx.x == 32) {
						for (int k = 0; k < MAXHBATCHSIZE; k++) {
							if (Res_H[k] != -1) {
								subH[blockIdx.x][level][++Num_H[level][0]] = Res_H[k];
								subHBits[blockIdx.x][level][Num_H[level][0]] = Res_HBits[k];
							}
						}
						// Num_H_Bits[level][0] += CalNumOfOne(&subHBits[blockIdx.x][level][1], Num_H[level][0]);
					}
				}
				// 计算最后一个batch
				for (int j = threadIdx.x - 32; j < last_batch; j += (THREADNUM / 2)) {
					Res_H[j] = notzero_id_dev_H[id_offset_dev_H[non_vertex_father[tid]] + j + batch_begin_H * MAXHBATCHSIZE];
					Res_HBits[j] = bits_dev_H[id_offset_dev_H[non_vertex_father[tid]] + j + batch_begin_H * MAXHBATCHSIZE];
				}
				IntersectionDevBitsOld3(&notzero_id_dev_H[id_offset_dev_H[non_vertex[tid]]], num_subH, Res_H, last_batch, &bits_dev_H[id_offset_dev_H[non_vertex[tid]]], Res_HBits);
				if (threadIdx.x == 32) {
					for (int k = 0; k < last_batch; k++) {
						if (Res_H[k] != -1) {
							subH[blockIdx.x][level][++Num_H[level][0]] = Res_H[k];
							subHBits[blockIdx.x][level][Num_H[level][0]] = Res_HBits[k];
						}
					}
					Num_H_Bits[level][0] = CalNumOfOne(&subHBits[blockIdx.x][level][1], Num_H[level][0]);
					// subHBits[blockIdx.x][level][0] = Num_H_Bits[level][0];
				}
			}
		}
		__syncthreads();
		if (Num_L_Bits[level][0] >= *q && Num_H_Bits[level][0] >= *p - level - 2) {
			__syncthreads();
			if (threadIdx.x == 0) {
				level++;
				top++;
			}
			__syncthreads();
		}
		__syncthreads();
		while (top != 0) {
			__syncthreads();
			unsigned batch_idx = batch_info[blockIdx.x][level][0];
			// __syncthreads();
			if (threadIdx.x == 0) {
				if (batch_idx == 0) {
					int size = Num_H_Bits[level - 1][next_k];
					SBits[blockIdx.x][level][0] = size;
					subHBits[blockIdx.x][level][0] = size;
					S[blockIdx.x][level][0] = Num_H[level - 1][next_k];
					subH[blockIdx.x][level][0] = Num_H[level - 1][next_k];
					batch_info[blockIdx.x][level][1] = size / MAXBATCHLEVELSIZE;
				}
				unsigned all_batch = batch_info[blockIdx.x][level][1];
				batch_size = (batch_idx < all_batch ? MAXBATCHLEVELSIZE : subHBits[blockIdx.x][level][0] - all_batch * MAXBATCHLEVELSIZE);
				/*if (size > 100) {
					printf("AA\n");
				}*/
				/*printf("%d-%d-%d-%d-%d-%d\n", blockIdx.x, tid, level, next_k, batch_idx, all_batch);
				if (tid == 329 && level == 6 && next_k == 0) {
					printf("A\n");
				}*/
				// printf("%d\n",next_k);
				// printf("A\n");
				
				int count_tmp = 0, flag_bit = 0;
				for (int j = stack_bitmap[blockIdx.x][level][0]; j < subH[blockIdx.x][level][0]; j++) {
					unsigned tmp_offset_bitmap = subH[blockIdx.x][level - 1][offset_H[blockIdx.x][level - 1][batch_info[blockIdx.x][level - 1][2]] + j + 1];
					unsigned bit_num = subHBits[blockIdx.x][level - 1][offset_H[blockIdx.x][level - 1][batch_info[blockIdx.x][level - 1][2]] + j + 1];
					for (int k = stack_bitmap[blockIdx.x][level][1]; k < 32; k++) {
						if ((bit_num & (1 << k)) != 0) {
							unsigned vertex_tmp = k + 32 * tmp_offset_bitmap;
							end_L[count_tmp] = id_offset_dev_L[vertex_tmp + 1];
							begin_L[count_tmp] = id_offset_dev_L[vertex_tmp];

							end_H[count_tmp] = id_offset_dev_H[vertex_tmp + 1];
							begin_H[count_tmp] = id_offset_dev_H[vertex_tmp];
							count_tmp++;
							if (count_tmp == batch_size) {
								stack_bitmap[blockIdx.x][level][0] = j;
								stack_bitmap[blockIdx.x][level][1] = k + 1;
								flag_bit = 1;
								break;
							}
						}
					}
					if (flag_bit == 1) break;
					else {
						stack_bitmap[blockIdx.x][level][1] = 0;
					}
				}
			}
			__syncthreads();
			if (threadIdx.x < 32) {
				unsigned inter_offset = batch_info[blockIdx.x][level - 1][2] + 1;
				unsigned inter_idx = offset_L[blockIdx.x][level - 1][inter_offset - 1];
				unsigned* begin_S = &S[blockIdx.x][level - 1][inter_idx + 1];
				unsigned* begin_SBits = &SBits[blockIdx.x][level - 1][inter_idx + 1];
				unsigned num_S = Num_L[level - 1][inter_offset - 1];
				// unsigned num_SBits = Num_L_Bits[level - 1][inter_offset - 1];
				for (int i = threadIdx.x; i < batch_size; i += (THREADNUM / 2)) {
					offset_L[blockIdx.x][level][i] = i * num_S;
				}
				if (batch_size * num_S < MAXSBATCHSIZE) {
					// 将S扩至size倍
					for (int i = 0; i < batch_size; i++) {
						for (int j = threadIdx.x; j < num_S; j += (THREADNUM / 2)) {
							Res_S[j + i * num_S] = begin_S[j];
							Res_SBits[j + i * num_S] = begin_SBits[j];
						}
					}
					IntersectionDevBits6(notzero_id_dev_L, begin_L, end_L, Res_S, num_S, batch_size, bits_dev_L, Res_SBits);
					for (int i = threadIdx.x; i < batch_size; i += (THREADNUM / 2)) {
						int res_offset = i * num_S;
						Num_L[level][i] = 0;
						for (int k = 0; k < num_S; k++) {
							if (Res_S[k + res_offset] != -1) {
								S[blockIdx.x][level][res_offset + (++Num_L[level][i])] = Res_S[k + res_offset];
								SBits[blockIdx.x][level][res_offset + (Num_L[level][i])] = Res_SBits[k + res_offset];
							}
						}
						Num_L_Bits[level][i] = CalNumOfOne(&SBits[blockIdx.x][level][res_offset + 1], Num_L[level][i]);
						// SBits[blockIdx.x][level][res_offset] = Num_L_Bits[level][i];
					}
					//if (threadIdx.x == 0) if (size * num_S > 2000) printf("AAAA: %d,%d\n", num_S, size);
				}
				else {
					for (int i = 0; i < batch_size; i++) {
						// IntersectionDevBitsOld2(&notzero_id_dev_L[begin_L[i]], end_L[i] - begin_L[i], begin_S, num_S, Res_S, &bits_dev_L[begin_L[i]], begin_SBits, Res_SBits);
						// if (threadIdx.x == 0) {
						// 	int res_offset = i * num_S;
						// 	Num_L[level][i] = 0;
						// 	for (int k = 0; k < num_S; k++) {
						// 		if (Res_S[k] != -1) {
						// 			S[blockIdx.x][level][res_offset + (++Num_L[level][i])] = Res_S[k];
						// 			SBits[blockIdx.x][level][res_offset + (Num_L[level][i])] = Res_SBits[k];
						// 		}
						// 	}
						// 	Num_L_Bits[level][i] = CalNumOfOne(&SBits[blockIdx.x][level][res_offset + 1], Num_L[level][i]);
						// }
						unsigned num_L = end_L[i] - begin_L[i];
						if (num_S < num_L) {
							unsigned batch_begin_S = num_S / MAXSBATCHSIZE;
							unsigned last_batch = num_S - batch_begin_S * MAXSBATCHSIZE;
							if (threadIdx.x == 0) {
								Num_L[level][i] = 0;
								// Num_L_Bits[level][0] = 0;
							}
							for (int m = 0; m < batch_begin_S; m++) {
								// 填充Res_S
								for (int j = threadIdx.x; j < MAXSBATCHSIZE; j += (THREADNUM / 2)) {
									// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + i * MAXSBATCHSIZE];
									Res_S[j] = begin_S[j + m * MAXSBATCHSIZE];
									Res_SBits[j] = begin_SBits[j + m * MAXSBATCHSIZE];
								}
								// 计算该Batch的交集
								// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, MAXSBATCHSIZE);
								IntersectionDevBitsOld3(&notzero_id_dev_L[begin_L[i]], num_L, Res_S, MAXSBATCHSIZE, &bits_dev_L[begin_L[i]], Res_SBits);
								if (threadIdx.x == 0) {
									int res_offset = i * num_S;
									for (int k = 0; k < MAXSBATCHSIZE; k++) {
										if (Res_S[k] != -1) {
											S[blockIdx.x][level][res_offset + (++Num_L[level][i])] = Res_S[k];
											SBits[blockIdx.x][level][res_offset + Num_L[level][i]] = Res_SBits[k];
										}
									}
								}
							}
							// 计算最后一个batch
							for (int j = threadIdx.x; j < last_batch; j += (THREADNUM / 2)) {
								// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + batch_begin_S * MAXSBATCHSIZE];
								Res_S[j] = begin_S[j + batch_begin_S * MAXSBATCHSIZE];
								Res_SBits[j] = begin_SBits[j + batch_begin_S * MAXSBATCHSIZE];
							}
							// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, last_batch);
							IntersectionDevBitsOld3(&notzero_id_dev_L[begin_L[i]], num_L, Res_S, last_batch, &bits_dev_L[begin_L[i]], Res_SBits);
							if (threadIdx.x == 0) {
								int res_offset = i * num_S;
								for (int k = 0; k < last_batch; k++) {
									if (Res_S[k] != -1) {
										S[blockIdx.x][level][res_offset + (++Num_L[level][i])] = Res_S[k];
										SBits[blockIdx.x][level][res_offset + (Num_L[level][i])] = Res_SBits[k];
									}
								}
								Num_L_Bits[level][i] = CalNumOfOne(&SBits[blockIdx.x][level][res_offset + 1], Num_L[level][i]);
								// SBits[blockIdx.x][level][res_offset] = Num_L_Bits[level][i];
							}
							// IntersectionDevBitsOld2(&notzero_id_dev_L[id_offset_dev_L[non_vertex_father[tid]]], num_L, &notzero_id_dev_L[id_offset_dev_L[non_vertex[tid]]], num_S, Res_S, &bits_dev_L[id_offset_dev_L[non_vertex_father[tid]]], &bits_dev_L[id_offset_dev_L[non_vertex[tid]]], Res_SBits);
						}
						else {
							// IntersectionDevBitsOld2(&notzero_id_dev_L[id_offset_dev_L[non_vertex[tid]]], num_S, &notzero_id_dev_L[id_offset_dev_L[non_vertex_father[tid]]], num_L, Res_S, &bits_dev_L[id_offset_dev_L[non_vertex[tid]]], &bits_dev_L[id_offset_dev_L[non_vertex_father[tid]]], Res_SBits);
							unsigned batch_begin_L = num_L / MAXSBATCHSIZE;
							unsigned last_batch = num_L - batch_begin_L * MAXSBATCHSIZE;
							if (threadIdx.x == 0) {
								Num_L[level][i] = 0;
							}
							for (int m = 0; m < batch_begin_L; m++) {
								// 填充Res_S
								for (int j = threadIdx.x; j < MAXSBATCHSIZE; j += (THREADNUM / 2)) {
									// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + i * MAXSBATCHSIZE];
									Res_S[j] = notzero_id_dev_L[begin_L[i] + j + m * MAXSBATCHSIZE];
									Res_SBits[j] = bits_dev_L[begin_L[i] + j + m * MAXSBATCHSIZE];
								}
								// 计算该Batch的交集
								// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, MAXSBATCHSIZE);
								IntersectionDevBitsOld3(begin_S, num_S, Res_S, MAXSBATCHSIZE, begin_SBits, Res_SBits);
								if (threadIdx.x == 0) {
									int res_offset = i * num_S;
									for (int k = 0; k < MAXSBATCHSIZE; k++) {
										if (Res_S[k] != -1) {
											S[blockIdx.x][level][res_offset + (++Num_L[level][i])] = Res_S[k];
											SBits[blockIdx.x][level][res_offset + (Num_L[level][i])] = Res_SBits[k];
										}
									}
									// Num_L_Bits[level][0] += CalNumOfOne(&SBits[blockIdx.x][level][1], Num_L[level][0]);
								}
							}
							// 计算最后一个batch
							for (int j = threadIdx.x; j < last_batch; j += (THREADNUM / 2)) {
								// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + batch_begin_S * MAXSBATCHSIZE];
								Res_S[j] = notzero_id_dev_L[begin_L[i] + j + batch_begin_L * MAXSBATCHSIZE];
								Res_SBits[j] = bits_dev_L[begin_L[i] + j + batch_begin_L * MAXSBATCHSIZE];
							}
							// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, last_batch);
							IntersectionDevBitsOld3(begin_S, num_S, Res_S, last_batch, begin_SBits, Res_SBits);
							if (threadIdx.x == 0) {
								int res_offset = i * num_S;
								for (int k = 0; k < last_batch; k++) {
									if (Res_S[k] != -1) {
										S[blockIdx.x][level][res_offset + (++Num_L[level][i])] = Res_S[k];
										SBits[blockIdx.x][level][res_offset + (Num_L[level][i])] = Res_SBits[k];
									}
								}
								Num_L_Bits[level][i] = CalNumOfOne(&SBits[blockIdx.x][level][res_offset + 1], Num_L[level][i]);
								// SBits[blockIdx.x][level][res_offset] = Num_L_Bits[level][i];
							}
						}
					}
				}
			}
			else {
				unsigned inter_offset = batch_info[blockIdx.x][level - 1][2] + 1;
				// inter_offset = (inter_offset == 0 ? 1 : inter_offset);
				unsigned inter_idx = offset_H[blockIdx.x][level - 1][inter_offset - 1];
				unsigned* begin_subH = &subH[blockIdx.x][level - 1][inter_idx + 1];
				unsigned* begin_subHBits = &subHBits[blockIdx.x][level - 1][inter_idx + 1];
				unsigned num_subH = Num_H[level - 1][inter_offset - 1];
				// unsigned num_subHBits = Num_H_Bits[level - 1][inter_offset - 1];
				for (int i = threadIdx.x - 32; i < batch_size; i += (THREADNUM / 2)) {
					offset_H[blockIdx.x][level][i] = i * num_subH;
				}
				if (batch_size * num_subH < MAXHBATCHSIZE) {
					for (int i = 0; i < batch_size; i++) {
						for (int j = threadIdx.x - 32; j < num_subH; j += (THREADNUM / 2)) {
							Res_H[j + i * num_subH] = begin_subH[j];
							Res_HBits[j + i * num_subH] = begin_subHBits[j];
						}
					}
					//IntersectionDev6(col_dev_H, begin_H, end_H, Res_H, num_subH, num_subH);
					IntersectionDevBits6(notzero_id_dev_H, begin_H, end_H, Res_H, num_subH, batch_size, bits_dev_H, Res_HBits);
					for (int i = threadIdx.x - 32; i < batch_size; i += (THREADNUM / 2)) {
						int res_offset = i * num_subH;
						Num_H[level][i] = 0;
						for (int k = 0; k < num_subH; k++) {
							if (Res_H[k + res_offset] != -1) {
								subH[blockIdx.x][level][res_offset + (++Num_H[level][i])] = Res_H[k + res_offset];
								subHBits[blockIdx.x][level][res_offset + (Num_H[level][i])] = Res_HBits[k + res_offset];
							}
						}
						Num_H_Bits[level][i] = CalNumOfOne(&subHBits[blockIdx.x][level][res_offset + 1], Num_H[level][i]);
					}
					//if (threadIdx.x == 32) if (size * num_subH > 2000) printf("BBBB: %d,%d\n", num_subH, size);
				}
				else {
					for (int i = 0; i < batch_size; i++) {
						//IntersectionDevOld2(&col_dev_H[begin_H[i]], end_H[i] - begin_H[i], begin_subH, num_subH, Res_H);
						// IntersectionDevBitsOld2(&notzero_id_dev_H[begin_H[i]], end_H[i] - begin_H[i], begin_H, num_subH, Res_H, &bits_dev_H[begin_H[i]], begin_subHBits, Res_HBits);
						// if (threadIdx.x == 32) {
						// 	int res_offset = i * num_subH;
						// 	Num_H[level][i] = 0;
						// 	for (int k = 0; k < num_subH; k++) {
						// 		if (Res_H[k] != -1) {
						// 			subH[blockIdx.x][level][res_offset + (++Num_H[level][i])] = Res_H[k];
						// 			subHBits[blockIdx.x][level][res_offset + (Num_H[level][i])] = Res_HBits[k];
						// 		}
						// 	}
						// 	Num_H_Bits[level][i] = CalNumOfOne(&subHBits[blockIdx.x][level][res_offset + 1], Num_H[level][i]);
						unsigned num_H = end_H[i] - begin_H[i];
						if (num_subH < num_H) {
							// IntersectionDevBitsOld2(&notzero_id_dev_H[id_offset_dev_H[non_vertex_father[tid]]], num_H, &notzero_id_dev_H[id_offset_dev_H[non_vertex[tid]]], num_subH, Res_H, &bits_dev_H[id_offset_dev_H[non_vertex_father[tid]]], &bits_dev_H[id_offset_dev_H[non_vertex[tid]]], Res_HBits);
							unsigned batch_begin_subH = num_subH / MAXHBATCHSIZE;
							unsigned last_batch = num_subH - batch_begin_subH * MAXHBATCHSIZE;
							if (threadIdx.x == 32) {
								Num_H[level][i] = 0;
								// Num_H_Bits[level][0] = 0;
							}
							for (int m = 0; m < batch_begin_subH; m++) {
								// 填充Res_S
								for (int j = threadIdx.x - 32; j < MAXHBATCHSIZE; j += (THREADNUM / 2)) {
									// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + i * MAXSBATCHSIZE];
									Res_H[j] = begin_subH[j + m * MAXHBATCHSIZE];
									Res_HBits[j] = begin_subHBits[j + m * MAXHBATCHSIZE];
								}
								// 计算该Batch的交集
								// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, MAXSBATCHSIZE);
								IntersectionDevBitsOld3(&notzero_id_dev_H[begin_H[i]], num_H, Res_H, MAXHBATCHSIZE, &bits_dev_H[begin_H[i]], Res_HBits);
								if (threadIdx.x == 32) {
									int res_offset = i * num_subH;
									for (int k = 0; k < MAXHBATCHSIZE; k++) {
										if (Res_H[k] != -1) {
											subH[blockIdx.x][level][res_offset + (++Num_H[level][i])] = Res_H[k];
											subHBits[blockIdx.x][level][res_offset + (Num_H[level][i])] = Res_HBits[k];
										}
									}
									// Num_H_Bits[level][0] += CalNumOfOne(&subHBits[blockIdx.x][level][1], Num_H[level][0]);
								}
							}
							// 计算最后一个batch
							for (int j = threadIdx.x - 32; j < last_batch; j += (THREADNUM / 2)) {
								// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + batch_begin_S * MAXSBATCHSIZE];
								Res_H[j] = begin_subH[j + batch_begin_subH * MAXHBATCHSIZE];
								Res_HBits[j] = begin_subHBits[j + batch_begin_subH * MAXHBATCHSIZE];
							}
							// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, last_batch);
							IntersectionDevBitsOld3(&notzero_id_dev_H[begin_H[i]], num_H, Res_H, last_batch, &bits_dev_H[begin_H[i]], Res_HBits);
							if (threadIdx.x == 32) {
								int res_offset = i * num_subH;
								for (int k = 0; k < last_batch; k++) {
									if (Res_H[k] != -1) {
										subH[blockIdx.x][level][res_offset + (++Num_H[level][i])] = Res_H[k];
										subHBits[blockIdx.x][level][res_offset + (Num_H[level][i])] = Res_HBits[k];
									}
								}
								Num_H_Bits[level][i] = CalNumOfOne(&subHBits[blockIdx.x][level][res_offset + 1], Num_H[level][i]);
							}
						}
						else {
							// IntersectionDevBitsOld2(&notzero_id_dev_H[id_offset_dev_H[non_vertex[tid]]], num_subH, &notzero_id_dev_H[id_offset_dev_H[non_vertex_father[tid]]], num_H, Res_H, &bits_dev_H[id_offset_dev_H[non_vertex[tid]]], &bits_dev_H[id_offset_dev_H[non_vertex_father[tid]]], Res_HBits);
							unsigned batch_begin_H = num_H / MAXHBATCHSIZE;
							unsigned last_batch = num_H - batch_begin_H * MAXHBATCHSIZE;
							if (threadIdx.x == 32) {
								Num_H[level][i] = 0;
							}
							for (int m = 0; m < batch_begin_H; m++) {
								// 填充Res_S
								for (int j = threadIdx.x - 32; j < MAXHBATCHSIZE; j += (THREADNUM / 2)) {
									Res_H[j] = notzero_id_dev_H[begin_H[i] + j + m * MAXHBATCHSIZE];
									Res_HBits[j] = bits_dev_H[begin_H[i] + j + m * MAXHBATCHSIZE];
								}
								// 计算该Batch的交集
								IntersectionDevBitsOld3(begin_subH, num_subH, Res_H, MAXHBATCHSIZE, begin_subHBits, Res_HBits);
								if (threadIdx.x == 32) {
									int res_offset = i * num_subH;
									for (int k = 0; k < MAXHBATCHSIZE; k++) {
										if (Res_H[k] != -1) {
											subH[blockIdx.x][level][res_offset + (++Num_H[level][i])] = Res_H[k];
											subHBits[blockIdx.x][level][res_offset + Num_H[level][i]] = Res_HBits[k];
										}
									}
									// Num_H_Bits[level][0] += CalNumOfOne(&subHBits[blockIdx.x][level][1], Num_H[level][0]);
								}
							}
							// 计算最后一个batch
							for (int j = threadIdx.x - 32; j < last_batch; j += (THREADNUM / 2)) {
								Res_H[j] = notzero_id_dev_H[begin_H[i] + j + batch_begin_H * MAXHBATCHSIZE];
								Res_HBits[j] = bits_dev_H[begin_H[i] + j + batch_begin_H * MAXHBATCHSIZE];
							}
							IntersectionDevBitsOld3(begin_subH, num_subH, Res_H, last_batch, begin_subHBits, Res_HBits);
							if (threadIdx.x == 32) {
								int res_offset = i * num_subH;
								for (int k = 0; k < last_batch; k++) {
									if (Res_H[k] != -1) {
										subH[blockIdx.x][level][res_offset + (++Num_H[level][i])] = Res_H[k];
										subHBits[blockIdx.x][level][res_offset + Num_H[level][i]] = Res_HBits[k];
									}
								}
								Num_H_Bits[level][i] = CalNumOfOne(&subHBits[blockIdx.x][level][res_offset + 1], Num_H[level][i]);
							}
						}
					}
				}
			}
			__syncthreads();
			if (level == *p - 2) {
				for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
					unsigned m = Num_L_Bits[level][i];
					if (m >= *q) {
						// printf("%d-%d,%d,%d,%d,%d,%d,%d-%d-%d-%d-%d\n", tid, stack[blockIdx.x][1], stack[blockIdx.x][2], stack[blockIdx.x][3], stack[blockIdx.x][4], stack[blockIdx.x][5], stack[blockIdx.x][6], i, threadIdx.x, m, batch_size, batch_idx);
						atomicAdd(count, OrderMulDev(m, *q));
					}
				}
				__syncthreads();
				if (threadIdx.x == 0) {
					//stack[blockIdx.x][top] = 0;
					stack[blockIdx.x][level] = 0;
				}
				__syncthreads();
				if (batch_info[blockIdx.x][level][0] == batch_info[blockIdx.x][level][1]) {
					if (threadIdx.x == 0) {
						batch_info[blockIdx.x][level][0] = 0;
						batch_info[blockIdx.x][level][1] = 0;
						batch_info[blockIdx.x][level][2] = 0;
						stack_bitmap[blockIdx.x][level][0] = 0;
						stack_bitmap[blockIdx.x][level][1] = 0;
						stack[blockIdx.x][level] = 0;
						top--;
						level--;
						unsigned all_batch_tmp = batch_info[blockIdx.x][level][1];
						batch_size = (batch_info[blockIdx.x][level][0] < all_batch_tmp ? MAXBATCHLEVELSIZE : subHBits[blockIdx.x][level][0] - all_batch_tmp * MAXBATCHLEVELSIZE);
					}
				}
				else {
					if (threadIdx.x == 0) {
						batch_info[blockIdx.x][level][0]++;
					}
					__syncthreads();
					continue;
				}
			}
			__syncthreads();
			if (threadIdx.x == 0) {
				while (top != 0) {
					int flag = 0;
					/*if (stack[blockIdx.x][1] == 3 && stack[blockIdx.x][2] == 12 && stack[blockIdx.x][3] == 7 && stack[blockIdx.x][4] == 6 && level == 5) {
						printf("A\n");
					}*/
					for (int k = stack[blockIdx.x][level]; k < batch_size; k++) {
						if (Num_L_Bits[level][k] >= *q && Num_H_Bits[level][k] >= *p - level - 2) {
							stack[blockIdx.x][level] = k + 1;
							next_k = k;
							batch_info[blockIdx.x][level][2] = k;
							flag = 1;
							top++;
							level++;
							break;
						}
					}
					if (flag == 0) {
						stack[blockIdx.x][level] = 0;
						if (batch_info[blockIdx.x][level][0] == batch_info[blockIdx.x][level][1]) {
							batch_info[blockIdx.x][level][0] = 0;
							batch_info[blockIdx.x][level][1] = 0;
							batch_info[blockIdx.x][level][2] = 0;
							stack_bitmap[blockIdx.x][level][0] = 0;
							stack_bitmap[blockIdx.x][level][1] = 0;
							stack[blockIdx.x][level] = 0;
							level--;
							top--;
							unsigned all_batch_tmp = batch_info[blockIdx.x][level][1];
							batch_size = (batch_info[blockIdx.x][level][0] < all_batch_tmp ? MAXBATCHLEVELSIZE : subHBits[blockIdx.x][level][0] - all_batch_tmp * MAXBATCHLEVELSIZE);
						}
						else {
							batch_info[blockIdx.x][level][0]++;
							break;
						}
					}
					else {
						break;
					}
				}
			}
			__syncthreads();
		}
		if (threadIdx.x == 0) {
			tid += gridDim.x;
		}
		__syncthreads();
	}
	/*if (threadIdx.x == 0) {
		printf("%d end\n", blockIdx.x);
	}*/
}

__global__ void findCliqueGPUNewStealworkBatch(unsigned* id_offset_dev_L, unsigned* notzero_id_dev_L, unsigned* bits_dev_L, unsigned* id_offset_dev_H, unsigned* notzero_id_dev_H, unsigned* bits_dev_H, unsigned long long* count, unsigned* p, unsigned* q, unsigned* Hsize, unsigned* non_vertex, unsigned* non_vertex_father) {
	__shared__ int top, level;
	__shared__ unsigned next_k, tid, batch_size;
	__shared__ unsigned Res_SBits[MAXSBATCHSIZE], Res_HBits[MAXHBATCHSIZE], begin_L[MAXBATCHLEVELSIZE], end_L[MAXBATCHLEVELSIZE], begin_H[MAXBATCHLEVELSIZE], end_H[MAXBATCHLEVELSIZE], Num_L[MAXSTACKSIZE][MAXBATCHLEVELSIZE], Num_H[MAXSTACKSIZE][MAXBATCHLEVELSIZE], Num_L_Bits[MAXSTACKSIZE][MAXBATCHLEVELSIZE], Num_H_Bits[MAXSTACKSIZE][MAXBATCHLEVELSIZE];
	__shared__ int Res_S[MAXSBATCHSIZE], Res_H[MAXHBATCHSIZE];
	__shared__ unsigned next_node_id, skip, min_gcl, idx_gcl;
	tid = blockIdx.x;
	// unsigned long long start, end, start1, end1, start2, end2, duration = 0, duration1 = 0, duration2 = 0;
	__syncthreads();
	while (tid < *Hsize) {
		__syncthreads();
		if (threadIdx.x == 0) {
			top = 0;
			level = 0;
			next_k = 0;
			stack[blockIdx.x][top] = 0;
			glockArray[blockIdx.x] = 1;
			printf("%d:%d\n", blockIdx.x, tid);
		}
		__syncthreads();
		if (threadIdx.x < 32) {
			unsigned num_S = id_offset_dev_L[non_vertex[tid] + 1] - id_offset_dev_L[non_vertex[tid]], num_L = id_offset_dev_L[non_vertex_father[tid] + 1] - id_offset_dev_L[non_vertex_father[tid]];
			if (num_S < num_L) {
				unsigned batch_begin_S = num_S / MAXSBATCHSIZE;
				unsigned last_batch = num_S - batch_begin_S * MAXSBATCHSIZE;
				if (threadIdx.x == 0) {
					Num_L[level][0] = 0;
					// Num_L_Bits[level][0] = 0;
				}
				for (int i = 0; i < batch_begin_S; i++) {
					// 填充Res_S
					for (int j = threadIdx.x; j < MAXSBATCHSIZE; j += (THREADNUM / 2)) {
						// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + i * MAXSBATCHSIZE];
						Res_S[j] = notzero_id_dev_L[id_offset_dev_L[non_vertex[tid]] + j + i * MAXSBATCHSIZE];
						Res_SBits[j] = bits_dev_L[id_offset_dev_L[non_vertex[tid]] + j + i * MAXSBATCHSIZE];
					}
					// 计算该Batch的交集
					// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, MAXSBATCHSIZE);
					IntersectionDevBitsOld3(&notzero_id_dev_L[id_offset_dev_L[non_vertex_father[tid]]], num_L, Res_S, MAXSBATCHSIZE, &bits_dev_L[id_offset_dev_L[non_vertex_father[tid]]], Res_SBits);
					if (threadIdx.x == 0) {
						for (int k = 0; k < MAXSBATCHSIZE; k++) {
							if (Res_S[k] != -1) {
								S[blockIdx.x][level][++Num_L[level][0]] = Res_S[k];
								SBits[blockIdx.x][level][Num_L[level][0]] = Res_SBits[k];
							}
						}
					}
				}
				// 计算最后一个batch
				for (int j = threadIdx.x; j < last_batch; j += (THREADNUM / 2)) {
					// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + batch_begin_S * MAXSBATCHSIZE];
					Res_S[j] = notzero_id_dev_L[id_offset_dev_L[non_vertex[tid]] + j + batch_begin_S * MAXSBATCHSIZE];
					Res_SBits[j] = bits_dev_L[id_offset_dev_L[non_vertex[tid]] + j + batch_begin_S * MAXSBATCHSIZE];
				}
				// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, last_batch);
				IntersectionDevBitsOld3(&notzero_id_dev_L[id_offset_dev_L[non_vertex_father[tid]]], num_L, Res_S, last_batch, &bits_dev_L[id_offset_dev_L[non_vertex_father[tid]]], Res_SBits);
				if (threadIdx.x == 0) {
					for (int k = 0; k < last_batch; k++) {
						if (Res_S[k] != -1) {
							S[blockIdx.x][level][++Num_L[level][0]] = Res_S[k];
							SBits[blockIdx.x][level][Num_L[level][0]] = Res_SBits[k];
						}
					}
					Num_L_Bits[level][0] = CalNumOfOne(&SBits[blockIdx.x][level][1], Num_L[level][0]);
					// SBits[blockIdx.x][level][0] = Num_L_Bits[level][0];
				}
				// IntersectionDevBitsOld2(&notzero_id_dev_L[id_offset_dev_L[non_vertex_father[tid]]], num_L, &notzero_id_dev_L[id_offset_dev_L[non_vertex[tid]]], num_S, Res_S, &bits_dev_L[id_offset_dev_L[non_vertex_father[tid]]], &bits_dev_L[id_offset_dev_L[non_vertex[tid]]], Res_SBits);
			}
			else {
				// IntersectionDevBitsOld2(&notzero_id_dev_L[id_offset_dev_L[non_vertex[tid]]], num_S, &notzero_id_dev_L[id_offset_dev_L[non_vertex_father[tid]]], num_L, Res_S, &bits_dev_L[id_offset_dev_L[non_vertex[tid]]], &bits_dev_L[id_offset_dev_L[non_vertex_father[tid]]], Res_SBits);
				unsigned batch_begin_L = num_L / MAXSBATCHSIZE;
				unsigned last_batch = num_L - batch_begin_L * MAXSBATCHSIZE;
				if (threadIdx.x == 0) {
					Num_L[level][0] = 0;
				}
				for (int i = 0; i < batch_begin_L; i++) {
					// 填充Res_S
					for (int j = threadIdx.x; j < MAXSBATCHSIZE; j += (THREADNUM / 2)) {
						// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + i * MAXSBATCHSIZE];
						Res_S[j] = notzero_id_dev_L[id_offset_dev_L[non_vertex_father[tid]] + j + i * MAXSBATCHSIZE];
						Res_SBits[j] = bits_dev_L[id_offset_dev_L[non_vertex_father[tid]] + j + i * MAXSBATCHSIZE];
					}
					// 计算该Batch的交集
					// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, MAXSBATCHSIZE);
					IntersectionDevBitsOld3(&notzero_id_dev_L[id_offset_dev_L[non_vertex[tid]]], num_S, Res_S, MAXSBATCHSIZE, &bits_dev_L[id_offset_dev_L[non_vertex[tid]]], Res_SBits);
					if (threadIdx.x == 0) {
						for (int k = 0; k < MAXSBATCHSIZE; k++) {
							if (Res_S[k] != -1) {
								S[blockIdx.x][level][++Num_L[level][0]] = Res_S[k];
								SBits[blockIdx.x][level][Num_L[level][0]] = Res_SBits[k];
							}
						}
						// Num_L_Bits[level][0] += CalNumOfOne(&SBits[blockIdx.x][level][1], Num_L[level][0]);
					}
				}
				// 计算最后一个batch
				for (int j = threadIdx.x; j < last_batch; j += (THREADNUM / 2)) {
					// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + batch_begin_S * MAXSBATCHSIZE];
					Res_S[j] = notzero_id_dev_L[id_offset_dev_L[non_vertex_father[tid]] + j + batch_begin_L * MAXSBATCHSIZE];
					Res_SBits[j] = bits_dev_L[id_offset_dev_L[non_vertex_father[tid]] + j + batch_begin_L * MAXSBATCHSIZE];
				}
				// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, last_batch);
				IntersectionDevBitsOld3(&notzero_id_dev_L[id_offset_dev_L[non_vertex[tid]]], num_S, Res_S, last_batch, &bits_dev_L[id_offset_dev_L[non_vertex[tid]]], Res_SBits);
				if (threadIdx.x == 0) {
					for (int k = 0; k < last_batch; k++) {
						if (Res_S[k] != -1) {
							S[blockIdx.x][level][++Num_L[level][0]] = Res_S[k];
							SBits[blockIdx.x][level][Num_L[level][0]] = Res_SBits[k];
						}
					}
					Num_L_Bits[level][0] = CalNumOfOne(&SBits[blockIdx.x][level][1], Num_L[level][0]);
					// SBits[blockIdx.x][level][0] = Num_L_Bits[level][0];
				}
			}
		}
		else {
			unsigned num_subH = id_offset_dev_H[non_vertex[tid] + 1] - id_offset_dev_H[non_vertex[tid]], num_H = id_offset_dev_H[non_vertex_father[tid] + 1] - id_offset_dev_H[non_vertex_father[tid]];
			if (num_subH < num_H) {
				// IntersectionDevBitsOld2(&notzero_id_dev_H[id_offset_dev_H[non_vertex_father[tid]]], num_H, &notzero_id_dev_H[id_offset_dev_H[non_vertex[tid]]], num_subH, Res_H, &bits_dev_H[id_offset_dev_H[non_vertex_father[tid]]], &bits_dev_H[id_offset_dev_H[non_vertex[tid]]], Res_HBits);
				unsigned batch_begin_subH = num_subH / MAXHBATCHSIZE;
				unsigned last_batch = num_subH - batch_begin_subH * MAXHBATCHSIZE;
				if (threadIdx.x == 32) {
					Num_H[level][0] = 0;
					// Num_H_Bits[level][0] = 0;
				}
				for (int i = 0; i < batch_begin_subH; i++) {
					// 填充Res_S
					for (int j = threadIdx.x - 32; j < MAXHBATCHSIZE; j += (THREADNUM / 2)) {
						// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + i * MAXSBATCHSIZE];
						Res_H[j] = notzero_id_dev_H[id_offset_dev_H[non_vertex[tid]] + j + i * MAXHBATCHSIZE];
						Res_HBits[j] = bits_dev_H[id_offset_dev_H[non_vertex[tid]] + j + i * MAXHBATCHSIZE];
					}
					// 计算该Batch的交集
					// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, MAXSBATCHSIZE);
					IntersectionDevBitsOld3(&notzero_id_dev_H[id_offset_dev_H[non_vertex_father[tid]]], num_H, Res_H, MAXHBATCHSIZE, &bits_dev_H[id_offset_dev_H[non_vertex_father[tid]]], Res_HBits);
					if (threadIdx.x == 32) {
						for (int k = 0; k < MAXHBATCHSIZE; k++) {
							if (Res_H[k] != -1) {
								subH[blockIdx.x][level][++Num_H[level][0]] = Res_H[k];
								subHBits[blockIdx.x][level][Num_H[level][0]] = Res_HBits[k];
							}
						}
						// Num_H_Bits[level][0] += CalNumOfOne(&subHBits[blockIdx.x][level][1], Num_H[level][0]);
					}
				}
				// 计算最后一个batch
				for (int j = threadIdx.x - 32; j < last_batch; j += (THREADNUM / 2)) {
					// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + batch_begin_S * MAXSBATCHSIZE];
					Res_H[j] = notzero_id_dev_H[id_offset_dev_H[non_vertex[tid]] + j + batch_begin_subH * MAXHBATCHSIZE];
					Res_HBits[j] = bits_dev_H[id_offset_dev_H[non_vertex[tid]] + j + batch_begin_subH * MAXHBATCHSIZE];
				}
				// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, last_batch);
				IntersectionDevBitsOld3(&notzero_id_dev_H[id_offset_dev_H[non_vertex_father[tid]]], num_H, Res_H, last_batch, &bits_dev_H[id_offset_dev_H[non_vertex_father[tid]]], Res_HBits);
				if (threadIdx.x == 32) {
					for (int k = 0; k < last_batch; k++) {
						if (Res_H[k] != -1) {
							subH[blockIdx.x][level][++Num_H[level][0]] = Res_H[k];
							subHBits[blockIdx.x][level][Num_H[level][0]] = Res_HBits[k];
						}
					}
					Num_H_Bits[level][0] = CalNumOfOne(&subHBits[blockIdx.x][level][1], Num_H[level][0]);
					// subHBits[blockIdx.x][level][0] = Num_H_Bits[level][0];
				}
			}
			else {
				// IntersectionDevBitsOld2(&notzero_id_dev_H[id_offset_dev_H[non_vertex[tid]]], num_subH, &notzero_id_dev_H[id_offset_dev_H[non_vertex_father[tid]]], num_H, Res_H, &bits_dev_H[id_offset_dev_H[non_vertex[tid]]], &bits_dev_H[id_offset_dev_H[non_vertex_father[tid]]], Res_HBits);
				unsigned batch_begin_H = num_H / MAXHBATCHSIZE;
				unsigned last_batch = num_H - batch_begin_H * MAXHBATCHSIZE;
				if (threadIdx.x == 32) {
					Num_H[level][0] = 0;
				}
				for (int i = 0; i < batch_begin_H; i++) {
					// 填充Res_S
					for (int j = threadIdx.x - 32; j < MAXHBATCHSIZE; j += (THREADNUM / 2)) {
						Res_H[j] = notzero_id_dev_H[id_offset_dev_H[non_vertex_father[tid]] + j + i * MAXHBATCHSIZE];
						Res_HBits[j] = bits_dev_H[id_offset_dev_H[non_vertex_father[tid]] + j + i * MAXHBATCHSIZE];
					}
					// 计算该Batch的交集
					IntersectionDevBitsOld3(&notzero_id_dev_H[id_offset_dev_H[non_vertex[tid]]], num_subH, Res_H, MAXHBATCHSIZE, &bits_dev_H[id_offset_dev_H[non_vertex[tid]]], Res_HBits);
					if (threadIdx.x == 32) {
						for (int k = 0; k < MAXHBATCHSIZE; k++) {
							if (Res_H[k] != -1) {
								subH[blockIdx.x][level][++Num_H[level][0]] = Res_H[k];
								subHBits[blockIdx.x][level][Num_H[level][0]] = Res_HBits[k];
							}
						}
						// Num_H_Bits[level][0] += CalNumOfOne(&subHBits[blockIdx.x][level][1], Num_H[level][0]);
					}
				}
				// 计算最后一个batch
				for (int j = threadIdx.x - 32; j < last_batch; j += (THREADNUM / 2)) {
					Res_H[j] = notzero_id_dev_H[id_offset_dev_H[non_vertex_father[tid]] + j + batch_begin_H * MAXHBATCHSIZE];
					Res_HBits[j] = bits_dev_H[id_offset_dev_H[non_vertex_father[tid]] + j + batch_begin_H * MAXHBATCHSIZE];
				}
				IntersectionDevBitsOld3(&notzero_id_dev_H[id_offset_dev_H[non_vertex[tid]]], num_subH, Res_H, last_batch, &bits_dev_H[id_offset_dev_H[non_vertex[tid]]], Res_HBits);
				if (threadIdx.x == 32) {
					for (int k = 0; k < last_batch; k++) {
						if (Res_H[k] != -1) {
							subH[blockIdx.x][level][++Num_H[level][0]] = Res_H[k];
							subHBits[blockIdx.x][level][Num_H[level][0]] = Res_HBits[k];
						}
					}
					Num_H_Bits[level][0] = CalNumOfOne(&subHBits[blockIdx.x][level][1], Num_H[level][0]);
					// subHBits[blockIdx.x][level][0] = Num_H_Bits[level][0];
				}
			}
		}
		__syncthreads();
		if (Num_L_Bits[level][0] >= *q && Num_H_Bits[level][0] >= *p - level - 2) {
			__syncthreads();
			if (threadIdx.x == 0) {
				level++;
				top++;
			}
			__syncthreads();
		}
		__syncthreads();
		while (top != 0) {
			__syncthreads();
			unsigned batch_idx = batch_info[blockIdx.x][level][0];
			// __syncthreads();
			if (threadIdx.x == 0) {
				if (batch_idx == 0) {
					int size = Num_H_Bits[level - 1][next_k];
					SBits[blockIdx.x][level][0] = size;
					subHBits[blockIdx.x][level][0] = size;
					S[blockIdx.x][level][0] = Num_H[level - 1][next_k];
					subH[blockIdx.x][level][0] = Num_H[level - 1][next_k];
					batch_info[blockIdx.x][level][1] = size / MAXBATCHLEVELSIZE;
				}
				unsigned all_batch = batch_info[blockIdx.x][level][1];
				batch_size = (batch_idx < all_batch ? MAXBATCHLEVELSIZE : subHBits[blockIdx.x][level][0] - all_batch * MAXBATCHLEVELSIZE);
				/*if (size > 100) {
					printf("AA\n");
				}*/
				/*printf("%d-%d-%d-%d-%d-%d\n", blockIdx.x, tid, level, next_k, batch_idx, all_batch);
				if (tid == 329 && level == 6 && next_k == 0) {
					printf("A\n");
				}*/
				// printf("%d\n",next_k);
				// printf("A\n");
				
				int count_tmp = 0, flag_bit = 0;
				for (int j = stack_bitmap[blockIdx.x][level][0]; j < subH[blockIdx.x][level][0]; j++) {
					unsigned tmp_offset_bitmap = subH[blockIdx.x][level - 1][offset_H[blockIdx.x][level - 1][batch_info[blockIdx.x][level - 1][2]] + j + 1];
					unsigned bit_num = subHBits[blockIdx.x][level - 1][offset_H[blockIdx.x][level - 1][batch_info[blockIdx.x][level - 1][2]] + j + 1];
					for (int k = stack_bitmap[blockIdx.x][level][1]; k < 32; k++) {
						if ((bit_num & (1 << k)) != 0) {
							unsigned vertex_tmp = k + 32 * tmp_offset_bitmap;
							end_L[count_tmp] = id_offset_dev_L[vertex_tmp + 1];
							begin_L[count_tmp] = id_offset_dev_L[vertex_tmp];

							end_H[count_tmp] = id_offset_dev_H[vertex_tmp + 1];
							begin_H[count_tmp] = id_offset_dev_H[vertex_tmp];
							count_tmp++;
							if (count_tmp == batch_size) {
								stack_bitmap[blockIdx.x][level][0] = j;
								stack_bitmap[blockIdx.x][level][1] = k + 1;
								flag_bit = 1;
								break;
							}
						}
					}
					if (flag_bit == 1) break;
					else {
						stack_bitmap[blockIdx.x][level][1] = 0;
					}
				}
			}
			__syncthreads();
			if (threadIdx.x < 32) {
				unsigned inter_offset = batch_info[blockIdx.x][level - 1][2] + 1;
				unsigned inter_idx = offset_L[blockIdx.x][level - 1][inter_offset - 1];
				unsigned* begin_S = &S[blockIdx.x][level - 1][inter_idx + 1];
				unsigned* begin_SBits = &SBits[blockIdx.x][level - 1][inter_idx + 1];
				unsigned num_S = Num_L[level - 1][inter_offset - 1];
				// unsigned num_SBits = Num_L_Bits[level - 1][inter_offset - 1];
				for (int i = threadIdx.x; i < batch_size; i += (THREADNUM / 2)) {
					offset_L[blockIdx.x][level][i] = i * num_S;
				}
				if (batch_size * num_S < MAXSBATCHSIZE) {
					// 将S扩至size倍
					for (int i = 0; i < batch_size; i++) {
						for (int j = threadIdx.x; j < num_S; j += (THREADNUM / 2)) {
							Res_S[j + i * num_S] = begin_S[j];
							Res_SBits[j + i * num_S] = begin_SBits[j];
						}
					}
					IntersectionDevBits6(notzero_id_dev_L, begin_L, end_L, Res_S, num_S, batch_size, bits_dev_L, Res_SBits);
					for (int i = threadIdx.x; i < batch_size; i += (THREADNUM / 2)) {
						int res_offset = i * num_S;
						Num_L[level][i] = 0;
						for (int k = 0; k < num_S; k++) {
							if (Res_S[k + res_offset] != -1) {
								S[blockIdx.x][level][res_offset + (++Num_L[level][i])] = Res_S[k + res_offset];
								SBits[blockIdx.x][level][res_offset + (Num_L[level][i])] = Res_SBits[k + res_offset];
							}
						}
						Num_L_Bits[level][i] = CalNumOfOne(&SBits[blockIdx.x][level][res_offset + 1], Num_L[level][i]);
						// SBits[blockIdx.x][level][res_offset] = Num_L_Bits[level][i];
					}
					//if (threadIdx.x == 0) if (size * num_S > 2000) printf("AAAA: %d,%d\n", num_S, size);
				}
				else {
					for (int i = 0; i < batch_size; i++) {
						// IntersectionDevBitsOld2(&notzero_id_dev_L[begin_L[i]], end_L[i] - begin_L[i], begin_S, num_S, Res_S, &bits_dev_L[begin_L[i]], begin_SBits, Res_SBits);
						// if (threadIdx.x == 0) {
						// 	int res_offset = i * num_S;
						// 	Num_L[level][i] = 0;
						// 	for (int k = 0; k < num_S; k++) {
						// 		if (Res_S[k] != -1) {
						// 			S[blockIdx.x][level][res_offset + (++Num_L[level][i])] = Res_S[k];
						// 			SBits[blockIdx.x][level][res_offset + (Num_L[level][i])] = Res_SBits[k];
						// 		}
						// 	}
						// 	Num_L_Bits[level][i] = CalNumOfOne(&SBits[blockIdx.x][level][res_offset + 1], Num_L[level][i]);
						// }
						unsigned num_L = end_L[i] - begin_L[i];
						if (num_S < num_L) {
							unsigned batch_begin_S = num_S / MAXSBATCHSIZE;
							unsigned last_batch = num_S - batch_begin_S * MAXSBATCHSIZE;
							if (threadIdx.x == 0) {
								Num_L[level][i] = 0;
								// Num_L_Bits[level][0] = 0;
							}
							for (int m = 0; m < batch_begin_S; m++) {
								// 填充Res_S
								for (int j = threadIdx.x; j < MAXSBATCHSIZE; j += (THREADNUM / 2)) {
									// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + i * MAXSBATCHSIZE];
									Res_S[j] = begin_S[j + m * MAXSBATCHSIZE];
									Res_SBits[j] = begin_SBits[j + m * MAXSBATCHSIZE];
								}
								// 计算该Batch的交集
								// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, MAXSBATCHSIZE);
								IntersectionDevBitsOld3(&notzero_id_dev_L[begin_L[i]], num_L, Res_S, MAXSBATCHSIZE, &bits_dev_L[begin_L[i]], Res_SBits);
								if (threadIdx.x == 0) {
									int res_offset = i * num_S;
									for (int k = 0; k < MAXSBATCHSIZE; k++) {
										if (Res_S[k] != -1) {
											S[blockIdx.x][level][res_offset + (++Num_L[level][i])] = Res_S[k];
											SBits[blockIdx.x][level][res_offset + Num_L[level][i]] = Res_SBits[k];
										}
									}
								}
							}
							// 计算最后一个batch
							for (int j = threadIdx.x; j < last_batch; j += (THREADNUM / 2)) {
								// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + batch_begin_S * MAXSBATCHSIZE];
								Res_S[j] = begin_S[j + batch_begin_S * MAXSBATCHSIZE];
								Res_SBits[j] = begin_SBits[j + batch_begin_S * MAXSBATCHSIZE];
							}
							// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, last_batch);
							IntersectionDevBitsOld3(&notzero_id_dev_L[begin_L[i]], num_L, Res_S, last_batch, &bits_dev_L[begin_L[i]], Res_SBits);
							if (threadIdx.x == 0) {
								int res_offset = i * num_S;
								for (int k = 0; k < last_batch; k++) {
									if (Res_S[k] != -1) {
										S[blockIdx.x][level][res_offset + (++Num_L[level][i])] = Res_S[k];
										SBits[blockIdx.x][level][res_offset + (Num_L[level][i])] = Res_SBits[k];
									}
								}
								Num_L_Bits[level][i] = CalNumOfOne(&SBits[blockIdx.x][level][res_offset + 1], Num_L[level][i]);
								// SBits[blockIdx.x][level][res_offset] = Num_L_Bits[level][i];
							}
							// IntersectionDevBitsOld2(&notzero_id_dev_L[id_offset_dev_L[non_vertex_father[tid]]], num_L, &notzero_id_dev_L[id_offset_dev_L[non_vertex[tid]]], num_S, Res_S, &bits_dev_L[id_offset_dev_L[non_vertex_father[tid]]], &bits_dev_L[id_offset_dev_L[non_vertex[tid]]], Res_SBits);
						}
						else {
							// IntersectionDevBitsOld2(&notzero_id_dev_L[id_offset_dev_L[non_vertex[tid]]], num_S, &notzero_id_dev_L[id_offset_dev_L[non_vertex_father[tid]]], num_L, Res_S, &bits_dev_L[id_offset_dev_L[non_vertex[tid]]], &bits_dev_L[id_offset_dev_L[non_vertex_father[tid]]], Res_SBits);
							unsigned batch_begin_L = num_L / MAXSBATCHSIZE;
							unsigned last_batch = num_L - batch_begin_L * MAXSBATCHSIZE;
							if (threadIdx.x == 0) {
								Num_L[level][i] = 0;
							}
							for (int m = 0; m < batch_begin_L; m++) {
								// 填充Res_S
								for (int j = threadIdx.x; j < MAXSBATCHSIZE; j += (THREADNUM / 2)) {
									// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + i * MAXSBATCHSIZE];
									Res_S[j] = notzero_id_dev_L[begin_L[i] + j + m * MAXSBATCHSIZE];
									Res_SBits[j] = bits_dev_L[begin_L[i] + j + m * MAXSBATCHSIZE];
								}
								// 计算该Batch的交集
								// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, MAXSBATCHSIZE);
								IntersectionDevBitsOld3(begin_S, num_S, Res_S, MAXSBATCHSIZE, begin_SBits, Res_SBits);
								if (threadIdx.x == 0) {
									int res_offset = i * num_S;
									for (int k = 0; k < MAXSBATCHSIZE; k++) {
										if (Res_S[k] != -1) {
											S[blockIdx.x][level][res_offset + (++Num_L[level][i])] = Res_S[k];
											SBits[blockIdx.x][level][res_offset + (Num_L[level][i])] = Res_SBits[k];
										}
									}
									// Num_L_Bits[level][0] += CalNumOfOne(&SBits[blockIdx.x][level][1], Num_L[level][0]);
								}
							}
							// 计算最后一个batch
							for (int j = threadIdx.x; j < last_batch; j += (THREADNUM / 2)) {
								// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + batch_begin_S * MAXSBATCHSIZE];
								Res_S[j] = notzero_id_dev_L[begin_L[i] + j + batch_begin_L * MAXSBATCHSIZE];
								Res_SBits[j] = bits_dev_L[begin_L[i] + j + batch_begin_L * MAXSBATCHSIZE];
							}
							// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, last_batch);
							IntersectionDevBitsOld3(begin_S, num_S, Res_S, last_batch, begin_SBits, Res_SBits);
							if (threadIdx.x == 0) {
								int res_offset = i * num_S;
								for (int k = 0; k < last_batch; k++) {
									if (Res_S[k] != -1) {
										S[blockIdx.x][level][res_offset + (++Num_L[level][i])] = Res_S[k];
										SBits[blockIdx.x][level][res_offset + (Num_L[level][i])] = Res_SBits[k];
									}
								}
								Num_L_Bits[level][i] = CalNumOfOne(&SBits[blockIdx.x][level][res_offset + 1], Num_L[level][i]);
								// SBits[blockIdx.x][level][res_offset] = Num_L_Bits[level][i];
							}
						}
					}
				}
			}
			else {
				unsigned inter_offset = batch_info[blockIdx.x][level - 1][2] + 1;
				// inter_offset = (inter_offset == 0 ? 1 : inter_offset);
				unsigned inter_idx = offset_H[blockIdx.x][level - 1][inter_offset - 1];
				unsigned* begin_subH = &subH[blockIdx.x][level - 1][inter_idx + 1];
				unsigned* begin_subHBits = &subHBits[blockIdx.x][level - 1][inter_idx + 1];
				unsigned num_subH = Num_H[level - 1][inter_offset - 1];
				// unsigned num_subHBits = Num_H_Bits[level - 1][inter_offset - 1];
				for (int i = threadIdx.x - 32; i < batch_size; i += (THREADNUM / 2)) {
					offset_H[blockIdx.x][level][i] = i * num_subH;
				}
				if (batch_size * num_subH < MAXHBATCHSIZE) {
					for (int i = 0; i < batch_size; i++) {
						for (int j = threadIdx.x - 32; j < num_subH; j += (THREADNUM / 2)) {
							Res_H[j + i * num_subH] = begin_subH[j];
							Res_HBits[j + i * num_subH] = begin_subHBits[j];
						}
					}
					//IntersectionDev6(col_dev_H, begin_H, end_H, Res_H, num_subH, num_subH);
					IntersectionDevBits6(notzero_id_dev_H, begin_H, end_H, Res_H, num_subH, batch_size, bits_dev_H, Res_HBits);
					for (int i = threadIdx.x - 32; i < batch_size; i += (THREADNUM / 2)) {
						int res_offset = i * num_subH;
						Num_H[level][i] = 0;
						for (int k = 0; k < num_subH; k++) {
							if (Res_H[k + res_offset] != -1) {
								subH[blockIdx.x][level][res_offset + (++Num_H[level][i])] = Res_H[k + res_offset];
								subHBits[blockIdx.x][level][res_offset + (Num_H[level][i])] = Res_HBits[k + res_offset];
							}
						}
						Num_H_Bits[level][i] = CalNumOfOne(&subHBits[blockIdx.x][level][res_offset + 1], Num_H[level][i]);
					}
					//if (threadIdx.x == 32) if (size * num_subH > 2000) printf("BBBB: %d,%d\n", num_subH, size);
				}
				else {
					for (int i = 0; i < batch_size; i++) {
						//IntersectionDevOld2(&col_dev_H[begin_H[i]], end_H[i] - begin_H[i], begin_subH, num_subH, Res_H);
						// IntersectionDevBitsOld2(&notzero_id_dev_H[begin_H[i]], end_H[i] - begin_H[i], begin_H, num_subH, Res_H, &bits_dev_H[begin_H[i]], begin_subHBits, Res_HBits);
						// if (threadIdx.x == 32) {
						// 	int res_offset = i * num_subH;
						// 	Num_H[level][i] = 0;
						// 	for (int k = 0; k < num_subH; k++) {
						// 		if (Res_H[k] != -1) {
						// 			subH[blockIdx.x][level][res_offset + (++Num_H[level][i])] = Res_H[k];
						// 			subHBits[blockIdx.x][level][res_offset + (Num_H[level][i])] = Res_HBits[k];
						// 		}
						// 	}
						// 	Num_H_Bits[level][i] = CalNumOfOne(&subHBits[blockIdx.x][level][res_offset + 1], Num_H[level][i]);
						unsigned num_H = end_H[i] - begin_H[i];
						if (num_subH < num_H) {
							// IntersectionDevBitsOld2(&notzero_id_dev_H[id_offset_dev_H[non_vertex_father[tid]]], num_H, &notzero_id_dev_H[id_offset_dev_H[non_vertex[tid]]], num_subH, Res_H, &bits_dev_H[id_offset_dev_H[non_vertex_father[tid]]], &bits_dev_H[id_offset_dev_H[non_vertex[tid]]], Res_HBits);
							unsigned batch_begin_subH = num_subH / MAXHBATCHSIZE;
							unsigned last_batch = num_subH - batch_begin_subH * MAXHBATCHSIZE;
							if (threadIdx.x == 32) {
								Num_H[level][i] = 0;
								// Num_H_Bits[level][0] = 0;
							}
							for (int m = 0; m < batch_begin_subH; m++) {
								// 填充Res_S
								for (int j = threadIdx.x - 32; j < MAXHBATCHSIZE; j += (THREADNUM / 2)) {
									// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + i * MAXSBATCHSIZE];
									Res_H[j] = begin_subH[j + m * MAXHBATCHSIZE];
									Res_HBits[j] = begin_subHBits[j + m * MAXHBATCHSIZE];
								}
								// 计算该Batch的交集
								// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, MAXSBATCHSIZE);
								IntersectionDevBitsOld3(&notzero_id_dev_H[begin_H[i]], num_H, Res_H, MAXHBATCHSIZE, &bits_dev_H[begin_H[i]], Res_HBits);
								if (threadIdx.x == 32) {
									int res_offset = i * num_subH;
									for (int k = 0; k < MAXHBATCHSIZE; k++) {
										if (Res_H[k] != -1) {
											subH[blockIdx.x][level][res_offset + (++Num_H[level][i])] = Res_H[k];
											subHBits[blockIdx.x][level][res_offset + (Num_H[level][i])] = Res_HBits[k];
										}
									}
									// Num_H_Bits[level][0] += CalNumOfOne(&subHBits[blockIdx.x][level][1], Num_H[level][0]);
								}
							}
							// 计算最后一个batch
							for (int j = threadIdx.x - 32; j < last_batch; j += (THREADNUM / 2)) {
								// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + batch_begin_S * MAXSBATCHSIZE];
								Res_H[j] = begin_subH[j + batch_begin_subH * MAXHBATCHSIZE];
								Res_HBits[j] = begin_subHBits[j + batch_begin_subH * MAXHBATCHSIZE];
							}
							// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, last_batch);
							IntersectionDevBitsOld3(&notzero_id_dev_H[begin_H[i]], num_H, Res_H, last_batch, &bits_dev_H[begin_H[i]], Res_HBits);
							if (threadIdx.x == 32) {
								int res_offset = i * num_subH;
								for (int k = 0; k < last_batch; k++) {
									if (Res_H[k] != -1) {
										subH[blockIdx.x][level][res_offset + (++Num_H[level][i])] = Res_H[k];
										subHBits[blockIdx.x][level][res_offset + (Num_H[level][i])] = Res_HBits[k];
									}
								}
								Num_H_Bits[level][i] = CalNumOfOne(&subHBits[blockIdx.x][level][res_offset + 1], Num_H[level][i]);
							}
						}
						else {
							// IntersectionDevBitsOld2(&notzero_id_dev_H[id_offset_dev_H[non_vertex[tid]]], num_subH, &notzero_id_dev_H[id_offset_dev_H[non_vertex_father[tid]]], num_H, Res_H, &bits_dev_H[id_offset_dev_H[non_vertex[tid]]], &bits_dev_H[id_offset_dev_H[non_vertex_father[tid]]], Res_HBits);
							unsigned batch_begin_H = num_H / MAXHBATCHSIZE;
							unsigned last_batch = num_H - batch_begin_H * MAXHBATCHSIZE;
							if (threadIdx.x == 32) {
								Num_H[level][i] = 0;
							}
							for (int m = 0; m < batch_begin_H; m++) {
								// 填充Res_S
								for (int j = threadIdx.x - 32; j < MAXHBATCHSIZE; j += (THREADNUM / 2)) {
									Res_H[j] = notzero_id_dev_H[begin_H[i] + j + m * MAXHBATCHSIZE];
									Res_HBits[j] = bits_dev_H[begin_H[i] + j + m * MAXHBATCHSIZE];
								}
								// 计算该Batch的交集
								IntersectionDevBitsOld3(begin_subH, num_subH, Res_H, MAXHBATCHSIZE, begin_subHBits, Res_HBits);
								if (threadIdx.x == 32) {
									int res_offset = i * num_subH;
									for (int k = 0; k < MAXHBATCHSIZE; k++) {
										if (Res_H[k] != -1) {
											subH[blockIdx.x][level][res_offset + (++Num_H[level][i])] = Res_H[k];
											subHBits[blockIdx.x][level][res_offset + Num_H[level][i]] = Res_HBits[k];
										}
									}
									// Num_H_Bits[level][0] += CalNumOfOne(&subHBits[blockIdx.x][level][1], Num_H[level][0]);
								}
							}
							// 计算最后一个batch
							for (int j = threadIdx.x - 32; j < last_batch; j += (THREADNUM / 2)) {
								Res_H[j] = notzero_id_dev_H[begin_H[i] + j + batch_begin_H * MAXHBATCHSIZE];
								Res_HBits[j] = bits_dev_H[begin_H[i] + j + batch_begin_H * MAXHBATCHSIZE];
							}
							IntersectionDevBitsOld3(begin_subH, num_subH, Res_H, last_batch, begin_subHBits, Res_HBits);
							if (threadIdx.x == 32) {
								int res_offset = i * num_subH;
								for (int k = 0; k < last_batch; k++) {
									if (Res_H[k] != -1) {
										subH[blockIdx.x][level][res_offset + (++Num_H[level][i])] = Res_H[k];
										subHBits[blockIdx.x][level][res_offset + Num_H[level][i]] = Res_HBits[k];
									}
								}
								Num_H_Bits[level][i] = CalNumOfOne(&subHBits[blockIdx.x][level][res_offset + 1], Num_H[level][i]);
							}
						}
					}
				}
			}
			__syncthreads();
			if (level == *p - 2) {
				for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
					unsigned m = Num_L_Bits[level][i];
					if (m >= *q) {
						// printf("%d-%d,%d,%d,%d,%d,%d,%d-%d-%d-%d-%d\n", tid, stack[blockIdx.x][1], stack[blockIdx.x][2], stack[blockIdx.x][3], stack[blockIdx.x][4], stack[blockIdx.x][5], stack[blockIdx.x][6], i, threadIdx.x, m, batch_size, batch_idx);
						atomicAdd(count, OrderMulDev(m, *q));
					}
				}
				__syncthreads();
				if (threadIdx.x == 0) {
					//stack[blockIdx.x][top] = 0;
					stack[blockIdx.x][level] = 0;
				}
				__syncthreads();
				if (batch_info[blockIdx.x][level][0] == batch_info[blockIdx.x][level][1]) {
					if (threadIdx.x == 0) {
						batch_info[blockIdx.x][level][0] = 0;
						batch_info[blockIdx.x][level][1] = 0;
						batch_info[blockIdx.x][level][2] = 0;
						stack_bitmap[blockIdx.x][level][0] = 0;
						stack_bitmap[blockIdx.x][level][1] = 0;
						stack[blockIdx.x][level] = 0;
						top--;
						level--;
						unsigned all_batch_tmp = batch_info[blockIdx.x][level][1];
						batch_size = (batch_info[blockIdx.x][level][0] < all_batch_tmp ? MAXBATCHLEVELSIZE : subHBits[blockIdx.x][level][0] - all_batch_tmp * MAXBATCHLEVELSIZE);
					}
				}
				else {
					if (threadIdx.x == 0) {
						batch_info[blockIdx.x][level][0]++;
					}
					__syncthreads();
					continue;
				}
			}
			__syncthreads();
			if (threadIdx.x == 0) {
				while (top != 0) {
					int flag = 0;
					/*if (stack[blockIdx.x][1] == 3 && stack[blockIdx.x][2] == 12 && stack[blockIdx.x][3] == 7 && stack[blockIdx.x][4] == 6 && level == 5) {
						printf("A\n");
					}*/
					for (int k = stack[blockIdx.x][level]; k < batch_size; k++) {
						if (Num_L_Bits[level][k] >= *q && Num_H_Bits[level][k] >= *p - level - 2) {
							stack[blockIdx.x][level] = k + 1;
							next_k = k;
							batch_info[blockIdx.x][level][2] = k;
							flag = 1;
							top++;
							level++;
							break;
						}
					}
					if (flag == 0) {
						stack[blockIdx.x][level] = 0;
						if (batch_info[blockIdx.x][level][0] == batch_info[blockIdx.x][level][1]) {
							batch_info[blockIdx.x][level][0] = 0;
							batch_info[blockIdx.x][level][1] = 0;
							batch_info[blockIdx.x][level][2] = 0;
							stack_bitmap[blockIdx.x][level][0] = 0;
							stack_bitmap[blockIdx.x][level][1] = 0;
							stack[blockIdx.x][level] = 0;
							level--;
							top--;
							unsigned all_batch_tmp = batch_info[blockIdx.x][level][1];
							batch_size = (batch_info[blockIdx.x][level][0] < all_batch_tmp ? MAXBATCHLEVELSIZE : subHBits[blockIdx.x][level][0] - all_batch_tmp * MAXBATCHLEVELSIZE);
						}
						else {
							batch_info[blockIdx.x][level][0]++;
							break;
						}
					}
					else {
						break;
					}
				}
			}
			__syncthreads();
		}
		__syncthreads();
		if (threadIdx.x == 0) {
			while (atomicExch(&glockArray[blockIdx.x], 0) == 0);
			// while (atomicCAS(&glockArray[blockIdx.x], 0, 1) != 0);
			unsigned tmp_gcl_ori = GCL[blockIdx.x];
			if (tmp_gcl_ori != 0xFFFFFFFF) {
				tid = (tmp_gcl_ori + 1) * BLOCKNUM + blockIdx.x;
				// printf("block:%d, GCL:%d\n", blockIdx.x, tmp_gcl_ori);
				// atomicAdd(&GCL[blockIdx.x], 1);
                GCL[blockIdx.x]++;
			}
			else {
				tid = 0xFFFFFFFF;
			}
			glockArray[blockIdx.x] = 1;
		}
		__syncthreads();
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		//printf("%d end %f s, %f s\n", blockIdx.x, (float)duration / 1000000, (float)duration1 / 1000000);
		// atomicExch(&GCL[blockIdx.x], 0xFFFFFFFF);
        GCL[blockIdx.x] = 0xFFFFFFFF;
		// printf("block %d starts to steal work\n", blockIdx.x);
	}
	// if(threadIdx.x == 0 || threadIdx.x == 32) printf("A block:%d, thread:%d\n",blockIdx.x,threadIdx.x);
	
	__syncthreads();
	while (true) {
		__syncthreads();
		if (threadIdx.x == 0) {
			min_gcl = 0xFFFFFFFF, idx_gcl = 0xFFFFFFFF, skip = 0;
			for (int i = ((blockIdx.x + 1) % BLOCKNUM); i != blockIdx.x; i = (i + 1) % BLOCKNUM) {
				unsigned tmp = GCL[i];
				if (tmp != 0xFFFFFFFF) {
					min_gcl = tmp;
					idx_gcl = i;
					break;
				}
			}
			printf("B");
		}
		__syncthreads();
		if (min_gcl == 0xFFFFFFFF) {
			__syncthreads();
			break;
		}
		__syncthreads();
		if(threadIdx.x == 0) {
			while (atomicExch(&glockArray[idx_gcl], 0) == 0);
            // while (atomicCAS(&glockArray[idx_gcl], 0, 1) != 0);
			next_node_id = (min_gcl + 1) * BLOCKNUM + idx_gcl;
			if (GCL[idx_gcl] != min_gcl) {
				skip = 1;
                // printf("A");
			}
			if (next_node_id >= *Hsize) {
				skip = 1;
				// atomicExch(&GCL[idx_gcl], 0xFFFFFFFF);
                GCL[idx_gcl] = 0xFFFFFFFF;
                // printf("A");
			}
			if (skip == 0) {
				atomicAdd(&GCL[idx_gcl], 1);
                // printf("A");
                // printf("B");
			}

			glockArray[idx_gcl] = 1;
            // atomicExch(&glockArray[idx_gcl], 1);
		}
		__syncthreads();
		if (skip == 0) {
			__syncthreads();
			if (threadIdx.x == 0) {
				top = 0;
				level = 0;
				next_k = 0;
				stack[blockIdx.x][top] = 0;
				printf("%d steal %d's %d\n", blockIdx.x, idx_gcl, next_node_id);
			}
			__syncthreads();
			if (threadIdx.x < 32) {
				unsigned num_S = id_offset_dev_L[non_vertex[next_node_id] + 1] - id_offset_dev_L[non_vertex[next_node_id]], num_L = id_offset_dev_L[non_vertex_father[next_node_id] + 1] - id_offset_dev_L[non_vertex_father[next_node_id]];
				if (num_S < num_L) {
					unsigned batch_begin_S = num_S / MAXSBATCHSIZE;
					unsigned last_batch = num_S - batch_begin_S * MAXSBATCHSIZE;
					if (threadIdx.x == 0) {
						Num_L[level][0] = 0;
						// Num_L_Bits[level][0] = 0;
					}
					for (int i = 0; i < batch_begin_S; i++) {
						// 填充Res_S
						for (int j = threadIdx.x; j < MAXSBATCHSIZE; j += (THREADNUM / 2)) {
							// Res_S[j] = col_dev_L[row_dev_L[non_vertex[next_node_id]] + j + i * MAXSBATCHSIZE];
							Res_S[j] = notzero_id_dev_L[id_offset_dev_L[non_vertex[next_node_id]] + j + i * MAXSBATCHSIZE];
							Res_SBits[j] = bits_dev_L[id_offset_dev_L[non_vertex[next_node_id]] + j + i * MAXSBATCHSIZE];
						}
						// 计算该Batch的交集
						// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[next_node_id]]], num_L, Res_S, MAXSBATCHSIZE);
						IntersectionDevBitsOld3(&notzero_id_dev_L[id_offset_dev_L[non_vertex_father[next_node_id]]], num_L, Res_S, MAXSBATCHSIZE, &bits_dev_L[id_offset_dev_L[non_vertex_father[next_node_id]]], Res_SBits);
						if (threadIdx.x == 0) {
							for (int k = 0; k < MAXSBATCHSIZE; k++) {
								if (Res_S[k] != -1) {
									S[blockIdx.x][level][++Num_L[level][0]] = Res_S[k];
									SBits[blockIdx.x][level][Num_L[level][0]] = Res_SBits[k];
								}
							}
						}
					}
					// 计算最后一个batch
					for (int j = threadIdx.x; j < last_batch; j += (THREADNUM / 2)) {
						// Res_S[j] = col_dev_L[row_dev_L[non_vertex[next_node_id]] + j + batch_begin_S * MAXSBATCHSIZE];
						Res_S[j] = notzero_id_dev_L[id_offset_dev_L[non_vertex[next_node_id]] + j + batch_begin_S * MAXSBATCHSIZE];
						Res_SBits[j] = bits_dev_L[id_offset_dev_L[non_vertex[next_node_id]] + j + batch_begin_S * MAXSBATCHSIZE];
					}
					// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[next_node_id]]], num_L, Res_S, last_batch);
					IntersectionDevBitsOld3(&notzero_id_dev_L[id_offset_dev_L[non_vertex_father[next_node_id]]], num_L, Res_S, last_batch, &bits_dev_L[id_offset_dev_L[non_vertex_father[next_node_id]]], Res_SBits);
					if (threadIdx.x == 0) {
						for (int k = 0; k < last_batch; k++) {
							if (Res_S[k] != -1) {
								S[blockIdx.x][level][++Num_L[level][0]] = Res_S[k];
								SBits[blockIdx.x][level][Num_L[level][0]] = Res_SBits[k];
							}
						}
						Num_L_Bits[level][0] = CalNumOfOne(&SBits[blockIdx.x][level][1], Num_L[level][0]);
						// SBits[blockIdx.x][level][0] = Num_L_Bits[level][0];
					}
					// IntersectionDevBitsOld2(&notzero_id_dev_L[id_offset_dev_L[non_vertex_father[next_node_id]]], num_L, &notzero_id_dev_L[id_offset_dev_L[non_vertex[next_node_id]]], num_S, Res_S, &bits_dev_L[id_offset_dev_L[non_vertex_father[next_node_id]]], &bits_dev_L[id_offset_dev_L[non_vertex[next_node_id]]], Res_SBits);
				}
				else {
					// IntersectionDevBitsOld2(&notzero_id_dev_L[id_offset_dev_L[non_vertex[next_node_id]]], num_S, &notzero_id_dev_L[id_offset_dev_L[non_vertex_father[next_node_id]]], num_L, Res_S, &bits_dev_L[id_offset_dev_L[non_vertex[next_node_id]]], &bits_dev_L[id_offset_dev_L[non_vertex_father[next_node_id]]], Res_SBits);
					unsigned batch_begin_L = num_L / MAXSBATCHSIZE;
					unsigned last_batch = num_L - batch_begin_L * MAXSBATCHSIZE;
					if (threadIdx.x == 0) {
						Num_L[level][0] = 0;
					}
					for (int i = 0; i < batch_begin_L; i++) {
						// 填充Res_S
						for (int j = threadIdx.x; j < MAXSBATCHSIZE; j += (THREADNUM / 2)) {
							// Res_S[j] = col_dev_L[row_dev_L[non_vertex[next_node_id]] + j + i * MAXSBATCHSIZE];
							Res_S[j] = notzero_id_dev_L[id_offset_dev_L[non_vertex_father[next_node_id]] + j + i * MAXSBATCHSIZE];
							Res_SBits[j] = bits_dev_L[id_offset_dev_L[non_vertex_father[next_node_id]] + j + i * MAXSBATCHSIZE];
						}
						// 计算该Batch的交集
						// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[next_node_id]]], num_L, Res_S, MAXSBATCHSIZE);
						IntersectionDevBitsOld3(&notzero_id_dev_L[id_offset_dev_L[non_vertex[next_node_id]]], num_S, Res_S, MAXSBATCHSIZE, &bits_dev_L[id_offset_dev_L[non_vertex[next_node_id]]], Res_SBits);
						if (threadIdx.x == 0) {
							for (int k = 0; k < MAXSBATCHSIZE; k++) {
								if (Res_S[k] != -1) {
									S[blockIdx.x][level][++Num_L[level][0]] = Res_S[k];
									SBits[blockIdx.x][level][Num_L[level][0]] = Res_SBits[k];
								}
							}
							// Num_L_Bits[level][0] += CalNumOfOne(&SBits[blockIdx.x][level][1], Num_L[level][0]);
						}
					}
					// 计算最后一个batch
					for (int j = threadIdx.x; j < last_batch; j += (THREADNUM / 2)) {
						// Res_S[j] = col_dev_L[row_dev_L[non_vertex[next_node_id]] + j + batch_begin_S * MAXSBATCHSIZE];
						Res_S[j] = notzero_id_dev_L[id_offset_dev_L[non_vertex_father[next_node_id]] + j + batch_begin_L * MAXSBATCHSIZE];
						Res_SBits[j] = bits_dev_L[id_offset_dev_L[non_vertex_father[next_node_id]] + j + batch_begin_L * MAXSBATCHSIZE];
					}
					// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[next_node_id]]], num_L, Res_S, last_batch);
					IntersectionDevBitsOld3(&notzero_id_dev_L[id_offset_dev_L[non_vertex[next_node_id]]], num_S, Res_S, last_batch, &bits_dev_L[id_offset_dev_L[non_vertex[next_node_id]]], Res_SBits);
					if (threadIdx.x == 0) {
						for (int k = 0; k < last_batch; k++) {
							if (Res_S[k] != -1) {
								S[blockIdx.x][level][++Num_L[level][0]] = Res_S[k];
								SBits[blockIdx.x][level][Num_L[level][0]] = Res_SBits[k];
							}
						}
						Num_L_Bits[level][0] = CalNumOfOne(&SBits[blockIdx.x][level][1], Num_L[level][0]);
						// SBits[blockIdx.x][level][0] = Num_L_Bits[level][0];
					}
				}
			}
			else {
				unsigned num_subH = id_offset_dev_H[non_vertex[next_node_id] + 1] - id_offset_dev_H[non_vertex[next_node_id]], num_H = id_offset_dev_H[non_vertex_father[next_node_id] + 1] - id_offset_dev_H[non_vertex_father[next_node_id]];
				if (num_subH < num_H) {
					// IntersectionDevBitsOld2(&notzero_id_dev_H[id_offset_dev_H[non_vertex_father[next_node_id]]], num_H, &notzero_id_dev_H[id_offset_dev_H[non_vertex[next_node_id]]], num_subH, Res_H, &bits_dev_H[id_offset_dev_H[non_vertex_father[next_node_id]]], &bits_dev_H[id_offset_dev_H[non_vertex[next_node_id]]], Res_HBits);
					unsigned batch_begin_subH = num_subH / MAXHBATCHSIZE;
					unsigned last_batch = num_subH - batch_begin_subH * MAXHBATCHSIZE;
					if (threadIdx.x == 32) {
						Num_H[level][0] = 0;
						// Num_H_Bits[level][0] = 0;
					}
					for (int i = 0; i < batch_begin_subH; i++) {
						// 填充Res_S
						for (int j = threadIdx.x - 32; j < MAXHBATCHSIZE; j += (THREADNUM / 2)) {
							// Res_S[j] = col_dev_L[row_dev_L[non_vertex[next_node_id]] + j + i * MAXSBATCHSIZE];
							Res_H[j] = notzero_id_dev_H[id_offset_dev_H[non_vertex[next_node_id]] + j + i * MAXHBATCHSIZE];
							Res_HBits[j] = bits_dev_H[id_offset_dev_H[non_vertex[next_node_id]] + j + i * MAXHBATCHSIZE];
						}
						// 计算该Batch的交集
						// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[next_node_id]]], num_L, Res_S, MAXSBATCHSIZE);
						IntersectionDevBitsOld3(&notzero_id_dev_H[id_offset_dev_H[non_vertex_father[next_node_id]]], num_H, Res_H, MAXHBATCHSIZE, &bits_dev_H[id_offset_dev_H[non_vertex_father[next_node_id]]], Res_HBits);
						if (threadIdx.x == 32) {
							for (int k = 0; k < MAXHBATCHSIZE; k++) {
								if (Res_H[k] != -1) {
									subH[blockIdx.x][level][++Num_H[level][0]] = Res_H[k];
									subHBits[blockIdx.x][level][Num_H[level][0]] = Res_HBits[k];
								}
							}
							// Num_H_Bits[level][0] += CalNumOfOne(&subHBits[blockIdx.x][level][1], Num_H[level][0]);
						}
					}
					// 计算最后一个batch
					for (int j = threadIdx.x - 32; j < last_batch; j += (THREADNUM / 2)) {
						// Res_S[j] = col_dev_L[row_dev_L[non_vertex[next_node_id]] + j + batch_begin_S * MAXSBATCHSIZE];
						Res_H[j] = notzero_id_dev_H[id_offset_dev_H[non_vertex[next_node_id]] + j + batch_begin_subH * MAXHBATCHSIZE];
						Res_HBits[j] = bits_dev_H[id_offset_dev_H[non_vertex[next_node_id]] + j + batch_begin_subH * MAXHBATCHSIZE];
					}
					// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[next_node_id]]], num_L, Res_S, last_batch);
					IntersectionDevBitsOld3(&notzero_id_dev_H[id_offset_dev_H[non_vertex_father[next_node_id]]], num_H, Res_H, last_batch, &bits_dev_H[id_offset_dev_H[non_vertex_father[next_node_id]]], Res_HBits);
					if (threadIdx.x == 32) {
						for (int k = 0; k < last_batch; k++) {
							if (Res_H[k] != -1) {
								subH[blockIdx.x][level][++Num_H[level][0]] = Res_H[k];
								subHBits[blockIdx.x][level][Num_H[level][0]] = Res_HBits[k];
							}
						}
						Num_H_Bits[level][0] = CalNumOfOne(&subHBits[blockIdx.x][level][1], Num_H[level][0]);
						// subHBits[blockIdx.x][level][0] = Num_H_Bits[level][0];
					}
				}
				else {
					// IntersectionDevBitsOld2(&notzero_id_dev_H[id_offset_dev_H[non_vertex[next_node_id]]], num_subH, &notzero_id_dev_H[id_offset_dev_H[non_vertex_father[next_node_id]]], num_H, Res_H, &bits_dev_H[id_offset_dev_H[non_vertex[next_node_id]]], &bits_dev_H[id_offset_dev_H[non_vertex_father[next_node_id]]], Res_HBits);
					unsigned batch_begin_H = num_H / MAXHBATCHSIZE;
					unsigned last_batch = num_H - batch_begin_H * MAXHBATCHSIZE;
					if (threadIdx.x == 32) {
						Num_H[level][0] = 0;
					}
					for (int i = 0; i < batch_begin_H; i++) {
						// 填充Res_S
						for (int j = threadIdx.x - 32; j < MAXHBATCHSIZE; j += (THREADNUM / 2)) {
							Res_H[j] = notzero_id_dev_H[id_offset_dev_H[non_vertex_father[next_node_id]] + j + i * MAXHBATCHSIZE];
							Res_HBits[j] = bits_dev_H[id_offset_dev_H[non_vertex_father[next_node_id]] + j + i * MAXHBATCHSIZE];
						}
						// 计算该Batch的交集
						IntersectionDevBitsOld3(&notzero_id_dev_H[id_offset_dev_H[non_vertex[next_node_id]]], num_subH, Res_H, MAXHBATCHSIZE, &bits_dev_H[id_offset_dev_H[non_vertex[next_node_id]]], Res_HBits);
						if (threadIdx.x == 32) {
							for (int k = 0; k < MAXHBATCHSIZE; k++) {
								if (Res_H[k] != -1) {
									subH[blockIdx.x][level][++Num_H[level][0]] = Res_H[k];
									subHBits[blockIdx.x][level][Num_H[level][0]] = Res_HBits[k];
								}
							}
							// Num_H_Bits[level][0] += CalNumOfOne(&subHBits[blockIdx.x][level][1], Num_H[level][0]);
						}
					}
					// 计算最后一个batch
					for (int j = threadIdx.x - 32; j < last_batch; j += (THREADNUM / 2)) {
						Res_H[j] = notzero_id_dev_H[id_offset_dev_H[non_vertex_father[next_node_id]] + j + batch_begin_H * MAXHBATCHSIZE];
						Res_HBits[j] = bits_dev_H[id_offset_dev_H[non_vertex_father[next_node_id]] + j + batch_begin_H * MAXHBATCHSIZE];
					}
					IntersectionDevBitsOld3(&notzero_id_dev_H[id_offset_dev_H[non_vertex[next_node_id]]], num_subH, Res_H, last_batch, &bits_dev_H[id_offset_dev_H[non_vertex[next_node_id]]], Res_HBits);
					if (threadIdx.x == 32) {
						for (int k = 0; k < last_batch; k++) {
							if (Res_H[k] != -1) {
								subH[blockIdx.x][level][++Num_H[level][0]] = Res_H[k];
								subHBits[blockIdx.x][level][Num_H[level][0]] = Res_HBits[k];
							}
						}
						Num_H_Bits[level][0] = CalNumOfOne(&subHBits[blockIdx.x][level][1], Num_H[level][0]);
						// subHBits[blockIdx.x][level][0] = Num_H_Bits[level][0];
					}
				}
			}
			__syncthreads();
			if (Num_L_Bits[level][0] >= *q && Num_H_Bits[level][0] >= *p - level - 2) {
				__syncthreads();
				if (threadIdx.x == 0) {
					level++;
					top++;
				}
				__syncthreads();
			}
			__syncthreads();
			while (top != 0) {
				__syncthreads();
				unsigned batch_idx = batch_info[blockIdx.x][level][0];
				// __syncthreads();
				if (threadIdx.x == 0) {
					if (batch_idx == 0) {
						int size = Num_H_Bits[level - 1][next_k];
						SBits[blockIdx.x][level][0] = size;
						subHBits[blockIdx.x][level][0] = size;
						S[blockIdx.x][level][0] = Num_H[level - 1][next_k];
						subH[blockIdx.x][level][0] = Num_H[level - 1][next_k];
						batch_info[blockIdx.x][level][1] = size / MAXBATCHLEVELSIZE;
					}
					unsigned all_batch = batch_info[blockIdx.x][level][1];
					batch_size = (batch_idx < all_batch ? MAXBATCHLEVELSIZE : subHBits[blockIdx.x][level][0] - all_batch * MAXBATCHLEVELSIZE);
					/*if (size > 100) {
						printf("AA\n");
					}*/
					/*printf("%d-%d-%d-%d-%d-%d\n", blockIdx.x, next_node_id, level, next_k, batch_idx, all_batch);
					if (next_node_id == 329 && level == 6 && next_k == 0) {
						printf("A\n");
					}*/
					// printf("%d\n",next_k);
					// printf("A\n");
					
					int count_tmp = 0, flag_bit = 0;
					for (int j = stack_bitmap[blockIdx.x][level][0]; j < subH[blockIdx.x][level][0]; j++) {
						unsigned tmp_offset_bitmap = subH[blockIdx.x][level - 1][offset_H[blockIdx.x][level - 1][batch_info[blockIdx.x][level - 1][2]] + j + 1];
						unsigned bit_num = subHBits[blockIdx.x][level - 1][offset_H[blockIdx.x][level - 1][batch_info[blockIdx.x][level - 1][2]] + j + 1];
						for (int k = stack_bitmap[blockIdx.x][level][1]; k < 32; k++) {
							if ((bit_num & (1 << k)) != 0) {
								unsigned vertex_tmp = k + 32 * tmp_offset_bitmap;
								end_L[count_tmp] = id_offset_dev_L[vertex_tmp + 1];
								begin_L[count_tmp] = id_offset_dev_L[vertex_tmp];

								end_H[count_tmp] = id_offset_dev_H[vertex_tmp + 1];
								begin_H[count_tmp] = id_offset_dev_H[vertex_tmp];
								count_tmp++;
								if (count_tmp == batch_size) {
									stack_bitmap[blockIdx.x][level][0] = j;
									stack_bitmap[blockIdx.x][level][1] = k + 1;
									flag_bit = 1;
									break;
								}
							}
						}
						if (flag_bit == 1) break;
						else {
							stack_bitmap[blockIdx.x][level][1] = 0;
						}
					}
				}
				__syncthreads();
				if (threadIdx.x < 32) {
					unsigned inter_offset = batch_info[blockIdx.x][level - 1][2] + 1;
					unsigned inter_idx = offset_L[blockIdx.x][level - 1][inter_offset - 1];
					unsigned* begin_S = &S[blockIdx.x][level - 1][inter_idx + 1];
					unsigned* begin_SBits = &SBits[blockIdx.x][level - 1][inter_idx + 1];
					unsigned num_S = Num_L[level - 1][inter_offset - 1];
					// unsigned num_SBits = Num_L_Bits[level - 1][inter_offset - 1];
					for (int i = threadIdx.x; i < batch_size; i += (THREADNUM / 2)) {
						offset_L[blockIdx.x][level][i] = i * num_S;
					}
					if (batch_size * num_S < MAXSBATCHSIZE) {
						// 将S扩至size倍
						for (int i = 0; i < batch_size; i++) {
							for (int j = threadIdx.x; j < num_S; j += (THREADNUM / 2)) {
								Res_S[j + i * num_S] = begin_S[j];
								Res_SBits[j + i * num_S] = begin_SBits[j];
							}
						}
						IntersectionDevBits6(notzero_id_dev_L, begin_L, end_L, Res_S, num_S, batch_size, bits_dev_L, Res_SBits);
						for (int i = threadIdx.x; i < batch_size; i += (THREADNUM / 2)) {
							int res_offset = i * num_S;
							Num_L[level][i] = 0;
							for (int k = 0; k < num_S; k++) {
								if (Res_S[k + res_offset] != -1) {
									S[blockIdx.x][level][res_offset + (++Num_L[level][i])] = Res_S[k + res_offset];
									SBits[blockIdx.x][level][res_offset + (Num_L[level][i])] = Res_SBits[k + res_offset];
								}
							}
							Num_L_Bits[level][i] = CalNumOfOne(&SBits[blockIdx.x][level][res_offset + 1], Num_L[level][i]);
							// SBits[blockIdx.x][level][res_offset] = Num_L_Bits[level][i];
						}
						//if (threadIdx.x == 0) if (size * num_S > 2000) printf("AAAA: %d,%d\n", num_S, size);
					}
					else {
						for (int i = 0; i < batch_size; i++) {
							// IntersectionDevBitsOld2(&notzero_id_dev_L[begin_L[i]], end_L[i] - begin_L[i], begin_S, num_S, Res_S, &bits_dev_L[begin_L[i]], begin_SBits, Res_SBits);
							// if (threadIdx.x == 0) {
							// 	int res_offset = i * num_S;
							// 	Num_L[level][i] = 0;
							// 	for (int k = 0; k < num_S; k++) {
							// 		if (Res_S[k] != -1) {
							// 			S[blockIdx.x][level][res_offset + (++Num_L[level][i])] = Res_S[k];
							// 			SBits[blockIdx.x][level][res_offset + (Num_L[level][i])] = Res_SBits[k];
							// 		}
							// 	}
							// 	Num_L_Bits[level][i] = CalNumOfOne(&SBits[blockIdx.x][level][res_offset + 1], Num_L[level][i]);
							// }
							unsigned num_L = end_L[i] - begin_L[i];
							if (num_S < num_L) {
								unsigned batch_begin_S = num_S / MAXSBATCHSIZE;
								unsigned last_batch = num_S - batch_begin_S * MAXSBATCHSIZE;
								if (threadIdx.x == 0) {
									Num_L[level][i] = 0;
									// Num_L_Bits[level][0] = 0;
								}
								for (int m = 0; m < batch_begin_S; m++) {
									// 填充Res_S
									for (int j = threadIdx.x; j < MAXSBATCHSIZE; j += (THREADNUM / 2)) {
										// Res_S[j] = col_dev_L[row_dev_L[non_vertex[next_node_id]] + j + i * MAXSBATCHSIZE];
										Res_S[j] = begin_S[j + m * MAXSBATCHSIZE];
										Res_SBits[j] = begin_SBits[j + m * MAXSBATCHSIZE];
									}
									// 计算该Batch的交集
									// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[next_node_id]]], num_L, Res_S, MAXSBATCHSIZE);
									IntersectionDevBitsOld3(&notzero_id_dev_L[begin_L[i]], num_L, Res_S, MAXSBATCHSIZE, &bits_dev_L[begin_L[i]], Res_SBits);
									if (threadIdx.x == 0) {
										int res_offset = i * num_S;
										for (int k = 0; k < MAXSBATCHSIZE; k++) {
											if (Res_S[k] != -1) {
												S[blockIdx.x][level][res_offset + (++Num_L[level][i])] = Res_S[k];
												SBits[blockIdx.x][level][res_offset + Num_L[level][i]] = Res_SBits[k];
											}
										}
									}
								}
								// 计算最后一个batch
								for (int j = threadIdx.x; j < last_batch; j += (THREADNUM / 2)) {
									// Res_S[j] = col_dev_L[row_dev_L[non_vertex[next_node_id]] + j + batch_begin_S * MAXSBATCHSIZE];
									Res_S[j] = begin_S[j + batch_begin_S * MAXSBATCHSIZE];
									Res_SBits[j] = begin_SBits[j + batch_begin_S * MAXSBATCHSIZE];
								}
								// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, last_batch);
								IntersectionDevBitsOld3(&notzero_id_dev_L[begin_L[i]], num_L, Res_S, last_batch, &bits_dev_L[begin_L[i]], Res_SBits);
								if (threadIdx.x == 0) {
									int res_offset = i * num_S;
									for (int k = 0; k < last_batch; k++) {
										if (Res_S[k] != -1) {
											S[blockIdx.x][level][res_offset + (++Num_L[level][i])] = Res_S[k];
											SBits[blockIdx.x][level][res_offset + (Num_L[level][i])] = Res_SBits[k];
										}
									}
									Num_L_Bits[level][i] = CalNumOfOne(&SBits[blockIdx.x][level][res_offset + 1], Num_L[level][i]);
									// SBits[blockIdx.x][level][res_offset] = Num_L_Bits[level][i];
								}
								// IntersectionDevBitsOld2(&notzero_id_dev_L[id_offset_dev_L[non_vertex_father[tid]]], num_L, &notzero_id_dev_L[id_offset_dev_L[non_vertex[tid]]], num_S, Res_S, &bits_dev_L[id_offset_dev_L[non_vertex_father[tid]]], &bits_dev_L[id_offset_dev_L[non_vertex[tid]]], Res_SBits);
							}
							else {
								// IntersectionDevBitsOld2(&notzero_id_dev_L[id_offset_dev_L[non_vertex[tid]]], num_S, &notzero_id_dev_L[id_offset_dev_L[non_vertex_father[tid]]], num_L, Res_S, &bits_dev_L[id_offset_dev_L[non_vertex[tid]]], &bits_dev_L[id_offset_dev_L[non_vertex_father[tid]]], Res_SBits);
								unsigned batch_begin_L = num_L / MAXSBATCHSIZE;
								unsigned last_batch = num_L - batch_begin_L * MAXSBATCHSIZE;
								if (threadIdx.x == 0) {
									Num_L[level][i] = 0;
								}
								for (int m = 0; m < batch_begin_L; m++) {
									// 填充Res_S
									for (int j = threadIdx.x; j < MAXSBATCHSIZE; j += (THREADNUM / 2)) {
										// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + i * MAXSBATCHSIZE];
										Res_S[j] = notzero_id_dev_L[begin_L[i] + j + m * MAXSBATCHSIZE];
										Res_SBits[j] = bits_dev_L[begin_L[i] + j + m * MAXSBATCHSIZE];
									}
									// 计算该Batch的交集
									// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, MAXSBATCHSIZE);
									IntersectionDevBitsOld3(begin_S, num_S, Res_S, MAXSBATCHSIZE, begin_SBits, Res_SBits);
									if (threadIdx.x == 0) {
										int res_offset = i * num_S;
										for (int k = 0; k < MAXSBATCHSIZE; k++) {
											if (Res_S[k] != -1) {
												S[blockIdx.x][level][res_offset + (++Num_L[level][i])] = Res_S[k];
												SBits[blockIdx.x][level][res_offset + (Num_L[level][i])] = Res_SBits[k];
											}
										}
										// Num_L_Bits[level][0] += CalNumOfOne(&SBits[blockIdx.x][level][1], Num_L[level][0]);
									}
								}
								// 计算最后一个batch
								for (int j = threadIdx.x; j < last_batch; j += (THREADNUM / 2)) {
									// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + batch_begin_S * MAXSBATCHSIZE];
									Res_S[j] = notzero_id_dev_L[begin_L[i] + j + batch_begin_L * MAXSBATCHSIZE];
									Res_SBits[j] = bits_dev_L[begin_L[i] + j + batch_begin_L * MAXSBATCHSIZE];
								}
								// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, last_batch);
								IntersectionDevBitsOld3(begin_S, num_S, Res_S, last_batch, begin_SBits, Res_SBits);
								if (threadIdx.x == 0) {
									int res_offset = i * num_S;
									for (int k = 0; k < last_batch; k++) {
										if (Res_S[k] != -1) {
											S[blockIdx.x][level][res_offset + (++Num_L[level][i])] = Res_S[k];
											SBits[blockIdx.x][level][res_offset + (Num_L[level][i])] = Res_SBits[k];
										}
									}
									Num_L_Bits[level][i] = CalNumOfOne(&SBits[blockIdx.x][level][res_offset + 1], Num_L[level][i]);
									// SBits[blockIdx.x][level][res_offset] = Num_L_Bits[level][i];
								}
							}
						}
					}
				}
				else {
					unsigned inter_offset = batch_info[blockIdx.x][level - 1][2] + 1;
					// inter_offset = (inter_offset == 0 ? 1 : inter_offset);
					unsigned inter_idx = offset_H[blockIdx.x][level - 1][inter_offset - 1];
					unsigned* begin_subH = &subH[blockIdx.x][level - 1][inter_idx + 1];
					unsigned* begin_subHBits = &subHBits[blockIdx.x][level - 1][inter_idx + 1];
					unsigned num_subH = Num_H[level - 1][inter_offset - 1];
					// unsigned num_subHBits = Num_H_Bits[level - 1][inter_offset - 1];
					for (int i = threadIdx.x - 32; i < batch_size; i += (THREADNUM / 2)) {
						offset_H[blockIdx.x][level][i] = i * num_subH;
					}
					if (batch_size * num_subH < MAXHBATCHSIZE) {
						for (int i = 0; i < batch_size; i++) {
							for (int j = threadIdx.x - 32; j < num_subH; j += (THREADNUM / 2)) {
								Res_H[j + i * num_subH] = begin_subH[j];
								Res_HBits[j + i * num_subH] = begin_subHBits[j];
							}
						}
						//IntersectionDev6(col_dev_H, begin_H, end_H, Res_H, num_subH, num_subH);
						IntersectionDevBits6(notzero_id_dev_H, begin_H, end_H, Res_H, num_subH, batch_size, bits_dev_H, Res_HBits);
						for (int i = threadIdx.x - 32; i < batch_size; i += (THREADNUM / 2)) {
							int res_offset = i * num_subH;
							Num_H[level][i] = 0;
							for (int k = 0; k < num_subH; k++) {
								if (Res_H[k + res_offset] != -1) {
									subH[blockIdx.x][level][res_offset + (++Num_H[level][i])] = Res_H[k + res_offset];
									subHBits[blockIdx.x][level][res_offset + (Num_H[level][i])] = Res_HBits[k + res_offset];
								}
							}
							Num_H_Bits[level][i] = CalNumOfOne(&subHBits[blockIdx.x][level][res_offset + 1], Num_H[level][i]);
						}
						//if (threadIdx.x == 32) if (size * num_subH > 2000) printf("BBBB: %d,%d\n", num_subH, size);
					}
					else {
						for (int i = 0; i < batch_size; i++) {
							//IntersectionDevOld2(&col_dev_H[begin_H[i]], end_H[i] - begin_H[i], begin_subH, num_subH, Res_H);
							// IntersectionDevBitsOld2(&notzero_id_dev_H[begin_H[i]], end_H[i] - begin_H[i], begin_H, num_subH, Res_H, &bits_dev_H[begin_H[i]], begin_subHBits, Res_HBits);
							// if (threadIdx.x == 32) {
							// 	int res_offset = i * num_subH;
							// 	Num_H[level][i] = 0;
							// 	for (int k = 0; k < num_subH; k++) {
							// 		if (Res_H[k] != -1) {
							// 			subH[blockIdx.x][level][res_offset + (++Num_H[level][i])] = Res_H[k];
							// 			subHBits[blockIdx.x][level][res_offset + (Num_H[level][i])] = Res_HBits[k];
							// 		}
							// 	}
							// 	Num_H_Bits[level][i] = CalNumOfOne(&subHBits[blockIdx.x][level][res_offset + 1], Num_H[level][i]);
							unsigned num_H = end_H[i] - begin_H[i];
							if (num_subH < num_H) {
								// IntersectionDevBitsOld2(&notzero_id_dev_H[id_offset_dev_H[non_vertex_father[tid]]], num_H, &notzero_id_dev_H[id_offset_dev_H[non_vertex[tid]]], num_subH, Res_H, &bits_dev_H[id_offset_dev_H[non_vertex_father[tid]]], &bits_dev_H[id_offset_dev_H[non_vertex[tid]]], Res_HBits);
								unsigned batch_begin_subH = num_subH / MAXHBATCHSIZE;
								unsigned last_batch = num_subH - batch_begin_subH * MAXHBATCHSIZE;
								if (threadIdx.x == 32) {
									Num_H[level][i] = 0;
									// Num_H_Bits[level][0] = 0;
								}
								for (int m = 0; m < batch_begin_subH; m++) {
									// 填充Res_S
									for (int j = threadIdx.x - 32; j < MAXHBATCHSIZE; j += (THREADNUM / 2)) {
										// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + i * MAXSBATCHSIZE];
										Res_H[j] = begin_subH[j + m * MAXHBATCHSIZE];
										Res_HBits[j] = begin_subHBits[j + m * MAXHBATCHSIZE];
									}
									// 计算该Batch的交集
									// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, MAXSBATCHSIZE);
									IntersectionDevBitsOld3(&notzero_id_dev_H[begin_H[i]], num_H, Res_H, MAXHBATCHSIZE, &bits_dev_H[begin_H[i]], Res_HBits);
									if (threadIdx.x == 32) {
										int res_offset = i * num_subH;
										for (int k = 0; k < MAXHBATCHSIZE; k++) {
											if (Res_H[k] != -1) {
												subH[blockIdx.x][level][res_offset + (++Num_H[level][i])] = Res_H[k];
												subHBits[blockIdx.x][level][res_offset + (Num_H[level][i])] = Res_HBits[k];
											}
										}
										// Num_H_Bits[level][0] += CalNumOfOne(&subHBits[blockIdx.x][level][1], Num_H[level][0]);
									}
								}
								// 计算最后一个batch
								for (int j = threadIdx.x - 32; j < last_batch; j += (THREADNUM / 2)) {
									// Res_S[j] = col_dev_L[row_dev_L[non_vertex[tid]] + j + batch_begin_S * MAXSBATCHSIZE];
									Res_H[j] = begin_subH[j + batch_begin_subH * MAXHBATCHSIZE];
									Res_HBits[j] = begin_subHBits[j + batch_begin_subH * MAXHBATCHSIZE];
								}
								// IntersectionDevOld3(&col_dev_L[row_dev_L[non_vertex_father[tid]]], num_L, Res_S, last_batch);
								IntersectionDevBitsOld3(&notzero_id_dev_H[begin_H[i]], num_H, Res_H, last_batch, &bits_dev_H[begin_H[i]], Res_HBits);
								if (threadIdx.x == 32) {
									int res_offset = i * num_subH;
									for (int k = 0; k < last_batch; k++) {
										if (Res_H[k] != -1) {
											subH[blockIdx.x][level][res_offset + (++Num_H[level][i])] = Res_H[k];
											subHBits[blockIdx.x][level][res_offset + (Num_H[level][i])] = Res_HBits[k];
										}
									}
									Num_H_Bits[level][i] = CalNumOfOne(&subHBits[blockIdx.x][level][res_offset + 1], Num_H[level][i]);
								}
							}
							else {
								// IntersectionDevBitsOld2(&notzero_id_dev_H[id_offset_dev_H[non_vertex[tid]]], num_subH, &notzero_id_dev_H[id_offset_dev_H[non_vertex_father[tid]]], num_H, Res_H, &bits_dev_H[id_offset_dev_H[non_vertex[tid]]], &bits_dev_H[id_offset_dev_H[non_vertex_father[tid]]], Res_HBits);
								unsigned batch_begin_H = num_H / MAXHBATCHSIZE;
								unsigned last_batch = num_H - batch_begin_H * MAXHBATCHSIZE;
								if (threadIdx.x == 32) {
									Num_H[level][i] = 0;
								}
								for (int m = 0; m < batch_begin_H; m++) {
									// 填充Res_S
									for (int j = threadIdx.x - 32; j < MAXHBATCHSIZE; j += (THREADNUM / 2)) {
										Res_H[j] = notzero_id_dev_H[begin_H[i] + j + m * MAXHBATCHSIZE];
										Res_HBits[j] = bits_dev_H[begin_H[i] + j + m * MAXHBATCHSIZE];
									}
									// 计算该Batch的交集
									IntersectionDevBitsOld3(begin_subH, num_subH, Res_H, MAXHBATCHSIZE, begin_subHBits, Res_HBits);
									if (threadIdx.x == 32) {
										int res_offset = i * num_subH;
										for (int k = 0; k < MAXHBATCHSIZE; k++) {
											if (Res_H[k] != -1) {
												subH[blockIdx.x][level][res_offset + (++Num_H[level][i])] = Res_H[k];
												subHBits[blockIdx.x][level][res_offset + Num_H[level][i]] = Res_HBits[k];
											}
										}
										// Num_H_Bits[level][0] += CalNumOfOne(&subHBits[blockIdx.x][level][1], Num_H[level][0]);
									}
								}
								// 计算最后一个batch
								for (int j = threadIdx.x - 32; j < last_batch; j += (THREADNUM / 2)) {
									Res_H[j] = notzero_id_dev_H[begin_H[i] + j + batch_begin_H * MAXHBATCHSIZE];
									Res_HBits[j] = bits_dev_H[begin_H[i] + j + batch_begin_H * MAXHBATCHSIZE];
								}
								IntersectionDevBitsOld3(begin_subH, num_subH, Res_H, last_batch, begin_subHBits, Res_HBits);
								if (threadIdx.x == 32) {
									int res_offset = i * num_subH;
									for (int k = 0; k < last_batch; k++) {
										if (Res_H[k] != -1) {
											subH[blockIdx.x][level][res_offset + (++Num_H[level][i])] = Res_H[k];
											subHBits[blockIdx.x][level][res_offset + Num_H[level][i]] = Res_HBits[k];
										}
									}
									Num_H_Bits[level][i] = CalNumOfOne(&subHBits[blockIdx.x][level][res_offset + 1], Num_H[level][i]);
								}
							}
						}
					}
				}
				__syncthreads();
				if (level == *p - 2) {
					for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
						unsigned m = Num_L_Bits[level][i];
						if (m >= *q) {
							// printf("%d-%d,%d,%d,%d,%d,%d,%d-%d-%d-%d-%d\n", tid, stack[blockIdx.x][1], stack[blockIdx.x][2], stack[blockIdx.x][3], stack[blockIdx.x][4], stack[blockIdx.x][5], stack[blockIdx.x][6], i, threadIdx.x, m, batch_size, batch_idx);
							atomicAdd(count, OrderMulDev(m, *q));
						}
					}
					__syncthreads();
					if (threadIdx.x == 0) {
						//stack[blockIdx.x][top] = 0;
						stack[blockIdx.x][level] = 0;
					}
					__syncthreads();
					if (batch_info[blockIdx.x][level][0] == batch_info[blockIdx.x][level][1]) {
						if (threadIdx.x == 0) {
							batch_info[blockIdx.x][level][0] = 0;
							batch_info[blockIdx.x][level][1] = 0;
							batch_info[blockIdx.x][level][2] = 0;
							stack_bitmap[blockIdx.x][level][0] = 0;
							stack_bitmap[blockIdx.x][level][1] = 0;
							stack[blockIdx.x][level] = 0;
							top--;
							level--;
							unsigned all_batch_tmp = batch_info[blockIdx.x][level][1];
							batch_size = (batch_info[blockIdx.x][level][0] < all_batch_tmp ? MAXBATCHLEVELSIZE : subHBits[blockIdx.x][level][0] - all_batch_tmp * MAXBATCHLEVELSIZE);
						}
					}
					else {
						if (threadIdx.x == 0) {
							batch_info[blockIdx.x][level][0]++;
						}
						__syncthreads();
						continue;
					}
				}
				__syncthreads();
				if (threadIdx.x == 0) {
					while (top != 0) {
						int flag = 0;
						/*if (stack[blockIdx.x][1] == 3 && stack[blockIdx.x][2] == 12 && stack[blockIdx.x][3] == 7 && stack[blockIdx.x][4] == 6 && level == 5) {
							printf("A\n");
						}*/
						for (int k = stack[blockIdx.x][level]; k < batch_size; k++) {
							if (Num_L_Bits[level][k] >= *q && Num_H_Bits[level][k] >= *p - level - 2) {
								stack[blockIdx.x][level] = k + 1;
								next_k = k;
								batch_info[blockIdx.x][level][2] = k;
								flag = 1;
								top++;
								level++;
								break;
							}
						}
						if (flag == 0) {
							stack[blockIdx.x][level] = 0;
							if (batch_info[blockIdx.x][level][0] == batch_info[blockIdx.x][level][1]) {
								batch_info[blockIdx.x][level][0] = 0;
								batch_info[blockIdx.x][level][1] = 0;
								batch_info[blockIdx.x][level][2] = 0;
								stack_bitmap[blockIdx.x][level][0] = 0;
								stack_bitmap[blockIdx.x][level][1] = 0;
								stack[blockIdx.x][level] = 0;
								level--;
								top--;
								unsigned all_batch_tmp = batch_info[blockIdx.x][level][1];
								batch_size = (batch_info[blockIdx.x][level][0] < all_batch_tmp ? MAXBATCHLEVELSIZE : subHBits[blockIdx.x][level][0] - all_batch_tmp * MAXBATCHLEVELSIZE);
							}
							else {
								batch_info[blockIdx.x][level][0]++;
								break;
							}
						}
						else {
							break;
						}
					}
				}
				__syncthreads();
			}
			__syncthreads();
		}
		__syncthreads();
	}
	__syncthreads();
}


int main(int argc, char* argv[]) {
	unsigned p = atoi(argv[2]), q = atoi(argv[3]);
    char* path = argv[1];
    CSR csrL;
    CSR csrH;
    Graph graphL;

    if(argc < 5){
        printf("Too few arguments\n");
        return 0;
    }
    else{
        printf("p: %s, q: %s, file: %s, left: %s\n", argv[2], argv[3], argv[1], argv[4]);
    }

    if (atoi(argv[4]) == 0) {
        printf("Select left to construct H\n");
		readFile(graphL, true, path);
		for (int i = 0; i < graphL.vertex_num; i++) {
			//printf("%d,", graphL.vertices[i].neighbor.size());
			sort(graphL.vertices[i].neighbor.begin(), graphL.vertices[i].neighbor.end());
		}
		std::cout << "vertexNum:" << graphL.vertex_num << "; edgeNum:" << graphL.edge_num << std::endl;


		Graph graphR;
		readFile(graphR, false, path);
		for (int i = 0; i < graphR.vertex_num; i++) {
			sort(graphR.vertices[i].neighbor.begin(), graphR.vertices[i].neighbor.end());
		}
		std::cout << "vertexNum:" << graphR.vertex_num << "; edgeNum:" << graphR.edge_num << std::endl;

		TrimGraphByCoreNew(graphL, graphR, p, q);
		std::cout << "vertexNum:" << graphL.vertex_num_after_trim << "; edgeNum:" << graphL.edge_num_after_trim << std::endl;
		std::cout << "vertexNum:" << graphR.vertex_num_after_trim << "; edgeNum:" << graphR.edge_num_after_trim << std::endl;
		
		// reformatGraph(graphL, graphR);
		// std::cout << "Reformat Done" << std::endl;

		Graph graphH;
		Collect2Hop(graphL, graphR, graphH, q);

		std::cout << "vertexNum:" << graphH.vertex_num << "; edgeNum:" << graphH.edge_num << std::endl;
		edgeDirectingByDegree(graphH);
		std::cout << "vertexNum:" << graphH.vertex_num << "; edgeNum:" << graphH.edge_num << std::endl;

		int zero_count = 0;
		std::vector<unsigned> nonzerovertex;
		for (int i = 0; i < graphH.vertex_num; i++) {
			if (graphH.vertices[i].neighbor.size() < p - 1) {
				zero_count++;
			}
			else {
				nonzerovertex.push_back(i);
			}
		}
		std::cout << "Number of degree-zero vertex: " << zero_count << "; Others: " << nonzerovertex.size() << std::endl;
		/*unsigned aa = nonzerovertex[158];
		nonzerovertex.clear();
		nonzerovertex.push_back(aa);*/
		/*for (int i = 0; i < nonzerovertex.size(); i++) {
			printf("%d,", graphH.vertices[nonzerovertex[i]].neighbor.size());

		}*/
		//******** New Stretagy ************
		std::vector<unsigned> firstlevel;
		std::vector<unsigned> firstlevelfather;
		for (int i = 0; i < nonzerovertex.size(); i++) {
			for (int j = 0; j < graphH.vertices[nonzerovertex[i]].neighbor.size(); j++) {
				firstlevel.push_back(graphH.vertices[nonzerovertex[i]].neighbor[j] - 1);
				firstlevelfather.push_back(nonzerovertex[i]);
			}
		}
		std::cout << "Number of first level: " << firstlevel.size() << std::endl;
		//**********************************
		graphL.transformToCSR(csrL);
		graphH.transformToCSR(csrH);

		BitMap bitmapL;
		BitMap bitmapH;
		graphL.transformToBitMap(bitmapL);
		graphH.transformToBitMap(bitmapH);

		//warm up GPU
		int* warmup = NULL;
		cudaMalloc(&warmup, sizeof(int));
		cudaFree(warmup);
		std::cout << "GPU warmup finished" << std::endl;

		unsigned long long count = 0;
		int H_size = firstlevel.size();;
		unsigned* id_offset_dev_L, * notzero_id_dev_L, * id_offset_dev_H, * notzero_id_dev_H, * bits_dev_H, * bits_dev_L, * non_vertex_dev, * non_vertex_dev_father;

		cudaMalloc((void**)&id_offset_dev_L, (graphL.vertex_num + 1) * sizeof(unsigned));
		cudaMalloc((void**)&notzero_id_dev_L, bitmapL.id_offset[graphL.vertex_num] * sizeof(unsigned));
		cudaMalloc((void**)&bits_dev_L, bitmapL.id_offset[graphL.vertex_num] * sizeof(unsigned));
		cudaMalloc((void**)&id_offset_dev_H, (graphH.vertex_num + 1) * sizeof(unsigned));
		cudaMalloc((void**)&notzero_id_dev_H, bitmapH.id_offset[graphH.vertex_num] * sizeof(unsigned));
		cudaMalloc((void**)&bits_dev_H, bitmapH.id_offset[graphH.vertex_num] * sizeof(unsigned));
		checkCudaErrors(cudaGetLastError());

		// cudaMalloc((void**)&non_vertex_dev, nonzerovertex.size() * sizeof(unsigned));
		// cudaMemcpy(non_vertex_dev, &nonzerovertex[0], nonzerovertex.size() * sizeof(unsigned), cudaMemcpyHostToDevice);
		//********* New Stretagy ******************
		cudaMalloc((void**)&non_vertex_dev, firstlevel.size() * sizeof(unsigned));
		cudaMemcpy(non_vertex_dev, &firstlevel[0], firstlevel.size() * sizeof(unsigned), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&non_vertex_dev_father, firstlevelfather.size() * sizeof(unsigned));
		cudaMemcpy(non_vertex_dev_father, &firstlevelfather[0], firstlevelfather.size() * sizeof(unsigned), cudaMemcpyHostToDevice);
		//*****************************************

		cudaMemcpy(id_offset_dev_L, bitmapL.id_offset, (graphL.vertex_num + 1) * sizeof(unsigned), cudaMemcpyHostToDevice);
		checkCudaErrors(cudaGetLastError());
		cudaMemcpy(notzero_id_dev_L, bitmapL.notzero_id, bitmapL.id_offset[graphL.vertex_num] * sizeof(unsigned), cudaMemcpyHostToDevice);
		checkCudaErrors(cudaGetLastError());
		cudaMemcpy(bits_dev_L, bitmapL.bitcode, bitmapL.id_offset[graphL.vertex_num] * sizeof(unsigned), cudaMemcpyHostToDevice);
		checkCudaErrors(cudaGetLastError());

		cudaMemcpy(id_offset_dev_H, bitmapH.id_offset, (graphH.vertex_num + 1) * sizeof(unsigned), cudaMemcpyHostToDevice);
		checkCudaErrors(cudaGetLastError());
		cudaMemcpy(notzero_id_dev_H, bitmapH.notzero_id, bitmapH.id_offset[graphH.vertex_num] * sizeof(unsigned), cudaMemcpyHostToDevice);
		checkCudaErrors(cudaGetLastError());
		cudaMemcpy(bits_dev_H, bitmapH.bitcode, bitmapH.id_offset[graphH.vertex_num] * sizeof(unsigned), cudaMemcpyHostToDevice);
		checkCudaErrors(cudaGetLastError());

		unsigned long long* count_dev;
		unsigned * p_dev, * q_dev, * H_size_dev;
		cudaMalloc((void**)&count_dev, sizeof(unsigned long long));
		cudaMalloc((void**)&q_dev, sizeof(unsigned));
		cudaMalloc((void**)&H_size_dev, sizeof(unsigned));
		cudaMalloc((void**)&p_dev, sizeof(unsigned));

		cudaMemcpy(count_dev, &count, sizeof(unsigned long long), cudaMemcpyHostToDevice);
		cudaMemcpy(q_dev, &q, sizeof(unsigned), cudaMemcpyHostToDevice);
		cudaMemcpy(p_dev, &p, sizeof(unsigned), cudaMemcpyHostToDevice);
		cudaMemcpy(H_size_dev, &H_size, sizeof(unsigned), cudaMemcpyHostToDevice);
		checkCudaErrors(cudaGetLastError());

		auto start = std::chrono::high_resolution_clock::now();
		findCliqueGPUNewStealworkBatch << <BLOCKNUM, THREADNUM >> > (id_offset_dev_L, notzero_id_dev_L, bits_dev_L, id_offset_dev_H, notzero_id_dev_H, bits_dev_H, count_dev, p_dev, q_dev, H_size_dev, non_vertex_dev, non_vertex_dev_father);
		checkCudaErrors(cudaGetLastError());
		cudaMemcpy(&count, count_dev, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
		checkCudaErrors(cudaGetLastError());
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> duration = end - start;
		std::cout << "\nAll time: " << duration.count() << "s" << std::endl;

		cudaFree(non_vertex_dev);

		cudaFree(id_offset_dev_L);
		cudaFree(notzero_id_dev_L);
		cudaFree(bits_dev_L);
		cudaFree(id_offset_dev_H);
		cudaFree(notzero_id_dev_H);
		cudaFree(bits_dev_H);
		cudaFree(count_dev);
		cudaFree(p_dev);
		cudaFree(q_dev);
		cudaFree(H_size_dev);
		//************* New Stretagy ****************
		cudaFree(non_vertex_dev_father);

		std::cout << "The number of (" << p << "," << q << ")-biclique is " << count << std::endl;
	}
	else{
		printf("Select right to construct H\n");
		readFile(graphL, false, path);
		for (int i = 0; i < graphL.vertex_num; i++) {
			//printf("%d,", graphL.vertices[i].neighbor.size());
			sort(graphL.vertices[i].neighbor.begin(), graphL.vertices[i].neighbor.end());
		}
		std::cout << "vertexNum:" << graphL.vertex_num << "; edgeNum:" << graphL.edge_num << std::endl;


		Graph graphR;
		readFile(graphR, true, path);
		for (int i = 0; i < graphR.vertex_num; i++) {
			sort(graphR.vertices[i].neighbor.begin(), graphR.vertices[i].neighbor.end());
		}
		std::cout << "vertexNum:" << graphR.vertex_num << "; edgeNum:" << graphR.edge_num << std::endl;

		TrimGraphByCoreNew(graphL, graphR, q, p);
		std::cout << "vertexNum:" << graphL.vertex_num_after_trim << "; edgeNum:" << graphL.edge_num_after_trim << std::endl;
		std::cout << "vertexNum:" << graphR.vertex_num_after_trim << "; edgeNum:" << graphR.edge_num_after_trim << std::endl;
		
		// reformatGraph(graphL, graphR);
		// std::cout << "Reformat Done" << std::endl;

		Graph graphH;
		Collect2Hop(graphL, graphR, graphH, p);

		std::cout << "vertexNum:" << graphH.vertex_num << "; edgeNum:" << graphH.edge_num << std::endl;
		edgeDirectingByDegree(graphH);
		std::cout << "vertexNum:" << graphH.vertex_num << "; edgeNum:" << graphH.edge_num << std::endl;

		int zero_count = 0;
		std::vector<unsigned> nonzerovertex;
		for (int i = 0; i < graphH.vertex_num; i++) {
			if (graphH.vertices[i].neighbor.size() < q - 1) {
				zero_count++;
			}
			else {
				nonzerovertex.push_back(i);
			}
		}
		std::cout << "Number of degree-zero vertex: " << zero_count << "; Others: " << nonzerovertex.size() << std::endl;
		/*unsigned aa = nonzerovertex[158];
		nonzerovertex.clear();
		nonzerovertex.push_back(aa);*/
		/*for (int i = 0; i < nonzerovertex.size(); i++) {
			printf("%d,", graphH.vertices[nonzerovertex[i]].neighbor.size());

		}*/
		//******** New Stretagy ************
		std::vector<unsigned> firstlevel;
		std::vector<unsigned> firstlevelfather;
		for (int i = 0; i < nonzerovertex.size(); i++) {
			for (int j = 0; j < graphH.vertices[nonzerovertex[i]].neighbor.size(); j++) {
				firstlevel.push_back(graphH.vertices[nonzerovertex[i]].neighbor[j] - 1);
				firstlevelfather.push_back(nonzerovertex[i]);
			}
		}
		std::cout << "Number of first level: " << firstlevel.size() << std::endl;
		//**********************************
		graphL.transformToCSR(csrL);
		graphH.transformToCSR(csrH);

		BitMap bitmapL;
		BitMap bitmapH;
		graphL.transformToBitMap(bitmapL);
		graphH.transformToBitMap(bitmapH);

		//warm up GPU
		int* warmup = NULL;
		cudaMalloc(&warmup, sizeof(int));
		cudaFree(warmup);
		std::cout << "GPU warmup finished" << std::endl;

		unsigned long long count = 0;
		int H_size = firstlevel.size();;
		unsigned* id_offset_dev_L, * notzero_id_dev_L, * id_offset_dev_H, * notzero_id_dev_H, * bits_dev_H, * bits_dev_L, * non_vertex_dev, * non_vertex_dev_father;

		cudaMalloc((void**)&id_offset_dev_L, (graphL.vertex_num + 1) * sizeof(unsigned));
		cudaMalloc((void**)&notzero_id_dev_L, bitmapL.id_offset[graphL.vertex_num] * sizeof(unsigned));
		cudaMalloc((void**)&bits_dev_L, bitmapL.id_offset[graphL.vertex_num] * sizeof(unsigned));
		cudaMalloc((void**)&id_offset_dev_H, (graphH.vertex_num + 1) * sizeof(unsigned));
		cudaMalloc((void**)&notzero_id_dev_H, bitmapH.id_offset[graphH.vertex_num] * sizeof(unsigned));
		cudaMalloc((void**)&bits_dev_H, bitmapH.id_offset[graphH.vertex_num] * sizeof(unsigned));
		checkCudaErrors(cudaGetLastError());

		// cudaMalloc((void**)&non_vertex_dev, nonzerovertex.size() * sizeof(unsigned));
		// cudaMemcpy(non_vertex_dev, &nonzerovertex[0], nonzerovertex.size() * sizeof(unsigned), cudaMemcpyHostToDevice);
		//********* New Stretagy ******************
		cudaMalloc((void**)&non_vertex_dev, firstlevel.size() * sizeof(unsigned));
		cudaMemcpy(non_vertex_dev, &firstlevel[0], firstlevel.size() * sizeof(unsigned), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&non_vertex_dev_father, firstlevelfather.size() * sizeof(unsigned));
		cudaMemcpy(non_vertex_dev_father, &firstlevelfather[0], firstlevelfather.size() * sizeof(unsigned), cudaMemcpyHostToDevice);
		//*****************************************

		cudaMemcpy(id_offset_dev_L, bitmapL.id_offset, (graphL.vertex_num + 1) * sizeof(unsigned), cudaMemcpyHostToDevice);
		checkCudaErrors(cudaGetLastError());
		cudaMemcpy(notzero_id_dev_L, bitmapL.notzero_id, bitmapL.id_offset[graphL.vertex_num] * sizeof(unsigned), cudaMemcpyHostToDevice);
		checkCudaErrors(cudaGetLastError());
		cudaMemcpy(bits_dev_L, bitmapL.bitcode, bitmapL.id_offset[graphL.vertex_num] * sizeof(unsigned), cudaMemcpyHostToDevice);
		checkCudaErrors(cudaGetLastError());

		cudaMemcpy(id_offset_dev_H, bitmapH.id_offset, (graphH.vertex_num + 1) * sizeof(unsigned), cudaMemcpyHostToDevice);
		checkCudaErrors(cudaGetLastError());
		cudaMemcpy(notzero_id_dev_H, bitmapH.notzero_id, bitmapH.id_offset[graphH.vertex_num] * sizeof(unsigned), cudaMemcpyHostToDevice);
		checkCudaErrors(cudaGetLastError());
		cudaMemcpy(bits_dev_H, bitmapH.bitcode, bitmapH.id_offset[graphH.vertex_num] * sizeof(unsigned), cudaMemcpyHostToDevice);
		checkCudaErrors(cudaGetLastError());

		unsigned long long* count_dev;
		unsigned* p_dev, * q_dev, * H_size_dev;
		cudaMalloc((void**)&count_dev, sizeof(unsigned long long));
		cudaMalloc((void**)&q_dev, sizeof(unsigned));
		cudaMalloc((void**)&H_size_dev, sizeof(unsigned));
		cudaMalloc((void**)&p_dev, sizeof(unsigned));

		cudaMemcpy(count_dev, &count, sizeof(unsigned long long), cudaMemcpyHostToDevice);
		cudaMemcpy(q_dev, &q, sizeof(unsigned), cudaMemcpyHostToDevice);
		cudaMemcpy(p_dev, &p, sizeof(unsigned), cudaMemcpyHostToDevice);
		cudaMemcpy(H_size_dev, &H_size, sizeof(unsigned), cudaMemcpyHostToDevice);
		checkCudaErrors(cudaGetLastError());

		auto start = std::chrono::high_resolution_clock::now();
		findCliqueGPUNewStealworkBatch << <BLOCKNUM, THREADNUM >> > (id_offset_dev_L, notzero_id_dev_L, bits_dev_L, id_offset_dev_H, notzero_id_dev_H, bits_dev_H, count_dev, q_dev, p_dev, H_size_dev, non_vertex_dev, non_vertex_dev_father);
		checkCudaErrors(cudaGetLastError());
		cudaMemcpy(&count, count_dev, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
		checkCudaErrors(cudaGetLastError());
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> duration = end - start;
		std::cout << "\nAll time: " << duration.count() << "s" << std::endl;

		cudaFree(non_vertex_dev);

		cudaFree(id_offset_dev_L);
		cudaFree(notzero_id_dev_L);
		cudaFree(bits_dev_L);
		cudaFree(id_offset_dev_H);
		cudaFree(notzero_id_dev_H);
		cudaFree(bits_dev_H);
		cudaFree(count_dev);
		cudaFree(p_dev);
		cudaFree(q_dev);
		cudaFree(H_size_dev);
		//************* New Stretagy ****************
		cudaFree(non_vertex_dev_father);

		std::cout << "The number of (" << p << "," << q << ")-biclique is " << count << std::endl;
	}
	//test();
	return 0;
}