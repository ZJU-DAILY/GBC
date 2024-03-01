#include<iostream>
#include<vector>
#include<map>
#include<queue>
#include<set>
#include <cstdlib>
#include<algorithm>
#include<string.h>
#include<stdlib.h>
#include<stdio.h>
#include<chrono>
#include"omp.h"

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

void TrimGraphByCore(Graph& L, Graph& R, int p, int q) {
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

		for (int j = 0; j < R.vertex_num; j++) {
			//printf("%d, ", j);
			if (R.vertices[j].label != 0) {
				if (left_removed.size() > 0) {
					if (left_removed[0] > R.vertices[j].neighbor[R.vertices[j].neighbor.size() - 1]) {
						continue;
					}
				}
				for (int i = 0; i < left_removed.size(); i++) {
					int it = BinarySearch(R.vertices[j].neighbor, R.vertices[j].neighbor.size(), left_removed[i]);
					if (it != -1) {
						R.vertices[j].neighbor.erase(R.vertices[j].neighbor.begin() + it);
						left_removed_edge++;
					}
				}
			}
		}
		printf("Now Left: %d\n", L.vertex_num_after_trim);

		L.edge_num_after_trim = L.edge_num_after_trim == 0 ? L.edge_num - left_removed_edge : L.edge_num_after_trim - left_removed_edge;

		//Right layer
		for (unsigned i = 0; i < R.vertex_num; i++) {
			if (R.vertices[i].label != 0 && R.vertices[i].neighbor.size() < p) {
				right_removed.push_back(i + 1);
				R.vertices[i].label = 0;
				count++;
			}
		}

		R.vertex_num_after_trim = R.vertex_num_after_trim == 0 ? R.vertex_num - right_removed.size() : R.vertex_num_after_trim - right_removed.size();

		for (int j = 0; j < L.vertex_num; j++) {
			if (L.vertices[j].label != 0) {
				if (right_removed.size() > 0) {
					if (right_removed[0] > L.vertices[j].neighbor[L.vertices[j].neighbor.size() - 1]) {
						continue;
					}
				}
				for (int i = 0; i < right_removed.size(); i++) {
					//auto it = find(L.vertices[j].neighbor.begin(), L.vertices[j].neighbor.end(), right_removed[i]);
					int it = BinarySearch(L.vertices[j].neighbor, L.vertices[j].neighbor.size(), right_removed[i]);
					if (it != -1) {
						L.vertices[j].neighbor.erase(L.vertices[j].neighbor.begin() + it);
						right_removed_edge++;
					}
				}
			}
		}
		printf("Now Right: %d\n", R.vertex_num_after_trim);

		R.edge_num_after_trim = R.edge_num_after_trim == 0 ? R.edge_num - right_removed_edge : R.edge_num_after_trim - right_removed_edge;
	}
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

void TrimGraphByCoreNewPlus(Graph& L, Graph& R, int p, int q) {
	std::vector<unsigned> left_removed, right_removed;
	unsigned right_removed_edge = 0, left_removed_edge = 0, end_left = 0, end_right = 0, start_left = 0, start_right = 0;
	for (unsigned i = 0; i < L.vertex_num; i++) {
		if (L.vertices[i].neighbor.size() < q) {
			left_removed.push_back(i + 1);
			end_left++;
			L.vertices[i].label = 0;
		}
	}
	for (unsigned i = 0; i < R.vertex_num; i++) {
		if (R.vertices[i].neighbor.size() < p) {
			right_removed.push_back(i + 1);
			end_right++;
			R.vertices[i].label = 0;
		}
	}
	while (start_left != end_left || start_right != end_right) {
		//for (int j = 0; j < R.vertex_num; j++) {
		if (start_left != end_left) {
			unsigned re_vertex = left_removed[start_left++];
			for (int j = 0; j < L.vertices[re_vertex - 1].neighbor.size(); j++) {
				unsigned vertex = L.vertices[re_vertex - 1].neighbor[j];
				int it = BinarySearch(R.vertices[vertex - 1].neighbor, R.vertices[vertex - 1].neighbor.size(), re_vertex);
				if (it != -1) {
					R.vertices[vertex - 1].neighbor.erase(R.vertices[vertex - 1].neighbor.begin() + it);
					left_removed_edge++;
					if (R.vertices[vertex - 1].neighbor.size() < p && R.vertices[vertex - 1].label != 0) {
						right_removed.push_back(vertex);
						end_right++;
						R.vertices[vertex - 1].label = 0;
					}
				}
			}
			L.vertices[re_vertex - 1].neighbor.clear();
		}
		if (start_right != end_right) {
			unsigned re_vertex = right_removed[start_right++];
			for (int j = 0; j < R.vertices[re_vertex - 1].neighbor.size(); j++) {
				unsigned vertex = R.vertices[re_vertex - 1].neighbor[j];
				//auto it = find(L.vertices[j].neighbor.begin(), L.vertices[j].neighbor.end(), right_removed[i]);
				int it = BinarySearch(L.vertices[vertex - 1].neighbor, L.vertices[vertex - 1].neighbor.size(), re_vertex);
				if (it != -1) {
					L.vertices[vertex - 1].neighbor.erase(L.vertices[vertex - 1].neighbor.begin() + it);
					right_removed_edge++;
					if (L.vertices[vertex - 1].neighbor.size() < q && L.vertices[vertex - 1].label != 0) {
						left_removed.push_back(vertex);
						end_left++;
						L.vertices[vertex - 1].label = 0;
					}
				}
			}
			R.vertices[re_vertex - 1].neighbor.clear();
		}
	}
	R.edge_num_after_trim = R.edge_num - right_removed_edge - left_removed_edge;
	L.edge_num_after_trim = L.edge_num - right_removed_edge - left_removed_edge;
	L.vertex_num_after_trim = L.vertex_num - left_removed.size();
	R.vertex_num_after_trim = R.vertex_num - right_removed.size();
}

bool cmp(std::pair<unsigned, unsigned> a, std::pair<unsigned, unsigned> b) {
	return a.second < b.second;
}

bool cmp_3(std::pair<unsigned, unsigned> a, std::pair<unsigned, unsigned> b) {
	return a.second > b.second;
}

void edgeDirectingByNeighborSize(Graph& H, Graph& G) {
	std::vector<std::pair<unsigned, unsigned>> lb_nsize;
	unsigned vertex_num = H.vertex_num;

	for (unsigned i = 0; i < vertex_num; i++) {
		lb_nsize.push_back(std::pair<unsigned, unsigned>(i + 1, G.vertices[i].neighbor.size()));
	}
	sort(lb_nsize.begin(), lb_nsize.end(), cmp_3); // sort by neighbor size in ascending order <vertex_id,neighbor_size>
	for (int j = 0; j < vertex_num; j++) {
		for (auto val : H.vertices[lb_nsize[j].first - 1].neighbor) {
			std::vector<unsigned>::iterator find_val = find(H.vertices[val - 1].neighbor.begin(), H.vertices[val - 1].neighbor.end(), lb_nsize[j].first);
			if (find_val != H.vertices[val - 1].neighbor.end()) {
				H.vertices[val - 1].neighbor.erase(find_val);
			}
		}
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
	printf("%d %d\n", count_L - 1, count_R - 1);
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
	printf("%d %d\n", count_L, count_R);
	L.vertex_num = L.vertex_num_after_trim;
	R.vertex_num = R.vertex_num_after_trim;
	L.edge_num = L.edge_num_after_trim;
	R.edge_num = R.edge_num_after_trim;
	dic_L.clear();
	dic_R.clear();
}

float disOfVertex(unsigned v, Graph& L, unsigned max_degree) {
	unsigned nei_size = L.vertices[v].neighbor.size(), res = 0;
	for (int i = 1; i < nei_size; i++) {
		unsigned a = L.vertices[v].neighbor[i] & 0x7FFFFFFF, b = L.vertices[v].neighbor[i - 1] & 0x7FFFFFFF;
		res += (a - b);
	}
	return (float)(res * nei_size) / max_degree;
	// return L.vertices[v].neighbor.size();
}

bool cmp_reorder(unsigned a, unsigned b) {
	return (a & 0x7FFFFFFF) < (b & 0x7FFFFFFF);
}

struct node {
	unsigned label;
	float dis;
	bool operator>(const node a)const
	{
		return dis < a.dis;
	}
};

void reorderVertex(Graph& L, Graph& R) {
	std::priority_queue<node, std::vector<node>, std::greater<node>> max_queue, max_queue_new;
	unsigned max_degree = 0;
	for(int i = 0; i < L.vertex_num; i++) {
		if (L.vertices[i].neighbor.size() > max_degree)  max_degree = L.vertices[i].neighbor.size();
	}
	for (unsigned i = 0; i < L.vertex_num; i++) {
		struct node v;
		v.label = i;
		v.dis = disOfVertex(i, L, max_degree);
		max_queue.push(v);
	}
	printf("%d\n", max_queue.top().label + 1);
	unsigned count = 1, all_num = 0;
	std::set<unsigned> reorder_batch, deleted_id;
	std::map<unsigned, unsigned> id_dic;
	while (!max_queue.empty()) {
		all_num++;
		struct node tmp = max_queue.top();
		if (all_num % 100 == 0) printf("%d\n", all_num);
		unsigned nei_num = L.vertices[tmp.label].neighbor.size();
		reorder_batch.clear();
		for (int i = 0; i < nei_num; i++) {
			unsigned to_write = L.vertices[tmp.label].neighbor[i];
			if ((to_write & 0x80000000) != 0) continue;
			id_dic[to_write] = count;
			reorder_batch.insert(to_write);
			L.vertices[tmp.label].neighbor[i] = count;
			count++;
		}
		deleted_id.insert(tmp.label);
		max_queue.pop();
		for (int i = 0; i < L.vertex_num; i++) {
			if (deleted_id.find(i) != deleted_id.end()) continue;
			int nei_num_tmp = L.vertices[i].neighbor.size();
			for (int j = 0; j < nei_num_tmp; j++) {
				unsigned nei_tmp = L.vertices[i].neighbor[j];
				if (reorder_batch.find(nei_tmp) != reorder_batch.end()) {
					L.vertices[i].neighbor[j] = id_dic[nei_tmp] | 0x80000000;
				}
			}
			sort(L.vertices[i].neighbor.begin(), L.vertices[i].neighbor.end(), cmp_reorder);
		}
		while (!max_queue.empty()) {
			struct node tmp_stu = max_queue.top();
			max_queue.pop();
			tmp_stu.dis = disOfVertex(tmp_stu.label, L, max_degree);
			max_queue_new.push(tmp_stu);
		}
		max_queue = max_queue_new;
		while (!max_queue_new.empty()) max_queue_new.pop();
	}
	std::map<unsigned, unsigned>::iterator iter; //定义迭代器 iter
	Graph Tmp_G = R;
	for (iter = id_dic.begin(); iter != id_dic.end(); ++iter) {
		//printf("%d->%d\n", iter->first, iter->second);
		//Vertex tmp_v = R.vertices[iter->first - 1];
		R.vertices[iter->second - 1] = Tmp_G.vertices[iter->first - 1];
		//R.vertices[iter->second - 1] = tmp_v;
		//R.vertices[iter->first - 1].label = iter->first;
		//R.vertices[iter->second - 1].label = iter->second;
	}
	printf("Reordered num: %d\n", count - 1);
	/*while (!max_queue.empty()) {
		struct node tmp = max_queue.top();
		printf("%u, %f; ", tmp.label, tmp.dis);
		max_queue.pop();
	}*/
}

int BinarySearchNew(std::vector<unsigned>& nums, int size, unsigned t) {
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

unsigned InterSectionBinary(std::vector<unsigned>& A, std::vector<unsigned>& B) {
	// size of B is smaller than size of A
	unsigned res_count = 0;
	int size = A.size();
	for (auto val : B) {
		int rt = BinarySearchNew(A, size, val);
		if (rt != -1) {
			res_count++;
		}
	}
	return res_count;
}

unsigned Dis(unsigned u, unsigned v, Graph& L){
	unsigned com_nei = InterSectionBinary(L.vertices[u].neighbor, L.vertices[v].neighbor);
	// if(u > v){
	// 	return com_nei * (u - v);
	// }
	// else{
	// 	return com_nei * (v - u);
	// }
	return com_nei;
}

unsigned sum_Dis(unsigned x, Graph& L, std::map<unsigned, unsigned>& dic, unsigned w, unsigned flag){
	unsigned all_now = dic.size(), begin = 0, res = 0;
	if(flag == 1) w = all_now % 32;
	if (w > all_now){
		w = all_now;
	}
	else{
		begin = all_now - w;
	}
	while(begin < all_now){
		res += Dis(x, dic[begin], L);
		begin++;
	}
	return res;
}

void reorderVertexNew(Graph& L, Graph& R, unsigned w, unsigned flag){
	clock_t start;
	std::map<unsigned, unsigned> order_dic;
	std::set<unsigned> remained_vertex;
	unsigned max_degree = 0, max_id = 0;
	for(unsigned i = 0; i < R.vertex_num; i++){
		remained_vertex.insert(i);
		if(R.vertices[i].neighbor.size() > max_degree){
			max_degree = R.vertices[i].neighbor.size();
			max_id = i;
		}
	}

	order_dic[0] = max_id;
	remained_vertex.erase(max_id);
	unsigned count = 1;
	start = clock();
	while(count < R.vertex_num){
		unsigned id_max = 0, dis_max = 0;
		for(unsigned x: remained_vertex){
			unsigned dis_this = sum_Dis(x, R, order_dic, w, flag);
			// if(remained_vertex.size() == 11) printf("%d\n", dis_this);
			if (dis_this >= dis_max){
				id_max = x;
				dis_max = dis_this;
			}
		}
		order_dic[count] = id_max;
		count++;
		remained_vertex.erase(id_max);
		if(count % 100 == 0)  printf("%d\n", count);
		// printf("%d -> %d : %d\n", id_max + 1, count, remained_vertex.size());
	}
	printf("Time: %.2f s\n", (float)(clock() - start) / CLOCKS_PER_SEC);
	std::map<unsigned, unsigned> reversed_dic;
	for(unsigned i = 0; i < order_dic.size(); i++){
		reversed_dic[order_dic[i]] = i;
	}

	for(int i = 0; i < L.vertex_num; i++){
		for(int j = 0; j < L.vertices[i].neighbor.size(); j++){
			L.vertices[i].neighbor[j] = reversed_dic[L.vertices[i].neighbor[j] - 1] + 1;
		}
	}

	Graph Tmp_R = R;
	for(unsigned i = 0; i < order_dic.size(); i++){
		R.vertices[i] = Tmp_R.vertices[order_dic[i]];
	}
}

bool cmp_reorderNeighbor(std::pair<unsigned, unsigned> a, std::pair<unsigned, unsigned> b) {
	return a.second > b.second;
}

void reorderVertexNewNeighbor(Graph& L, Graph& R){
	std::map<unsigned, unsigned> order_dic;
	std::vector<std::pair<unsigned, unsigned>> lb_degree;
	unsigned vertex_num = R.vertex_num;
	//int count = 0;
	for (unsigned i = 0; i < vertex_num; i++) {
		lb_degree.push_back(std::pair<unsigned, unsigned>(i, R.vertices[i].neighbor.size()));
	}
	sort(lb_degree.begin(), lb_degree.end(), cmp_reorderNeighbor);

	for(int i = 0; i < vertex_num; i++){
		order_dic[i] = lb_degree[i].first;
	}

	std::map<unsigned, unsigned> reversed_dic;
	for(unsigned i = 0; i < order_dic.size(); i++){
		reversed_dic[order_dic[i]] = i;
	}

	for(int i = 0; i < L.vertex_num; i++){
		for(int j = 0; j < L.vertices[i].neighbor.size(); j++){
			L.vertices[i].neighbor[j] = reversed_dic[L.vertices[i].neighbor[j] - 1] + 1;
		}
	}

	Graph Tmp_R = R;
	for(unsigned i = 0; i < order_dic.size(); i++){
		R.vertices[i] = Tmp_R.vertices[order_dic[i]];
	}
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

void reorderVertexH(Graph& R, unsigned w, Graph& L, unsigned flag){
	std::map<unsigned, unsigned> order_dic;
	std::set<unsigned> remained_vertex;
	Graph Tmp_R = R;

	unsigned max_degree = 0, max_id = 0;
	for(unsigned i = 0; i < R.vertex_num; i++){
		remained_vertex.insert(i);
		if(R.vertices[i].neighbor.size() > max_degree){
			max_degree = R.vertices[i].neighbor.size();
			max_id = i;
		}
	}

	Graph inR;
	for(int i = 0; i < R.vertex_num; i++){
		inR.addVertex(i + 1);
	}
	for(int i = 0; i < R.vertex_num; i++){
		for(int j = 0; j < R.vertices[i].neighbor.size(); j++){
			inR.addBipartiteEdge(R.vertices[i].neighbor[j], i + 1);
		}
	}

	order_dic[0] = max_id;
	remained_vertex.erase(max_id);
	unsigned count = 1;
	while(count < R.vertex_num){
		unsigned id_max = 0, dis_max = 0;
		for(unsigned x: remained_vertex){
			unsigned dis_this = sum_Dis(x, inR, order_dic, w, flag);
			// if(remained_vertex.size() == 11) printf("%d\n", dis_this);
			if (dis_this >= dis_max){
				id_max = x;
				dis_max = dis_this;
			}
		}
		order_dic[count] = id_max;
		count++;
		remained_vertex.erase(id_max);
		if(count % 100 == 0)  printf("%d\n", count);
		// printf("%d -> %d : %d\n", id_max + 1, count, remained_vertex.size());
	}

	std::map<unsigned, unsigned> reversed_dic;
	for(unsigned i = 0; i < order_dic.size(); i++){
		reversed_dic[order_dic[i]] = i;
	}

	for(int i = 0; i < Tmp_R.vertex_num; i++){
		for(int j = 0; j < Tmp_R.vertices[i].neighbor.size(); j++){
			Tmp_R.vertices[i].neighbor[j] = reversed_dic[Tmp_R.vertices[i].neighbor[j] - 1] + 1;
		}
	}

	
	for(unsigned i = 0; i < order_dic.size(); i++){
		R.vertices[i] = Tmp_R.vertices[order_dic[i]];
	}

	Graph Tmp_L = L;
	for(unsigned i = 0; i < order_dic.size(); i++){
		L.vertices[i] = Tmp_L.vertices[order_dic[i]];
	}
}

void calAllDis(Graph& L, Graph& R){
	float res_L = 0.0, res_R = 0.0;
	for(int j = 0; j < L.vertex_num; j++){
		unsigned nei_size = L.vertices[j].neighbor.size(), res_tmp = 0;
		for (int i = 1; i < nei_size; i++) {
			res_tmp += (L.vertices[j].neighbor[i] - L.vertices[j].neighbor[i - 1]);
		}
		res_L += (float) res_tmp / nei_size;
	}

	for(int j = 0; j < R.vertex_num; j++){
		unsigned nei_size = R.vertices[j].neighbor.size(), res_tmp = 0;
		for (int i = 1; i < nei_size; i++) {
			res_tmp += (R.vertices[j].neighbor[i] - R.vertices[j].neighbor[i - 1]);
		}
		res_R += (float) res_tmp / nei_size;
	}
	
	printf("Left: %f, Right: %f\n", res_L / L.vertex_num, res_R / R.vertex_num);
}

unsigned calQofVertex(Graph& L, unsigned v, Graph& R) {
	unsigned res_num = 0;
	for (auto x : R.vertices[v - 1].neighbor) {
		// search for v
		int res_id = BinarySearchNew(L.vertices[x - 1].neighbor, L.vertices[x - 1].neighbor.size(), v);
		if (res_id == -1) continue;
		if (v % 32 == 0) {
			unsigned left = v - 31;
			res_id--;
			while (res_id >= 0) {
				if (L.vertices[x - 1].neighbor[res_id] >= left) {
					res_num++;
					res_id--;
				}
				else break;
			}
		}
		else {
			unsigned left = (v / 32) * 32 + 1;
			unsigned right = left + 31;
			int res_id_left = res_id - 1, res_id_right = res_id + 1;
			while (res_id_left >= 0) {
				if (L.vertices[x - 1].neighbor[res_id_left] >= left) {
					res_num++;
					res_id_left--;
				}
				else break;
			}
			while (res_id_right < L.vertices[x - 1].neighbor.size()) {
				if (L.vertices[x - 1].neighbor[res_id_right] <= right) {
					res_num++;
					res_id_right++;
				}
				else break;
			}
		}
	}
	return res_num;
}

void reorderForBitmap(Graph& L, Graph& R) {
	std::map<unsigned, unsigned> vertex_Q;
	std::set<unsigned> remained_vertex;
	std::map<unsigned, unsigned> reorder_dic;
	unsigned max_id = 0, max_Q = 0;
	for (int i = 0; i < R.vertex_num_after_trim; i++) {
		remained_vertex.insert(i);
		vertex_Q[i] = calQofVertex(L, i + 1, R);
	}
	while (!remained_vertex.empty()) {
		max_id = 0xFFFFFFFF;
		max_Q = 0;
		for (auto x : remained_vertex) {
			if (vertex_Q[x] >= max_Q) {
				max_Q = vertex_Q[x];
				max_id = x;
			}
		}
		if (max_id == 0xFFFFFFFF) continue;
		remained_vertex.erase(max_id);
		if (remained_vertex.size() % 100 == 0) {
			printf("%ld\n", remained_vertex.size());
		}
		// 寻找令max_id的Q值增加最多的节点（stre: 1. 令max_id增加最多; 2. 找最小Q节点，放到Q增加最多的节点中; 3. 令max_id的Q增加量与该节点Q的减小量差值最大）
		unsigned max_nei_id = 0xFFFFFFFF, max_nei = 0;
		unsigned left = (max_id / 32) * 32 + 1;
		for (auto x : remained_vertex) {
			if (x + 1 >= left && x <= left + 30) continue;
			unsigned com_nei = InterSectionBinary(R.vertices[max_id].neighbor, R.vertices[x].neighbor);
			if (com_nei > max_nei) {
				max_nei_id = x;
				max_nei = com_nei;
			}
		}

		if (max_nei_id == 0xFFFFFFFF) continue;

		// 统计max_id所在块各个位的1的个数
		unsigned bit_num[32] = { 0 };
		for (auto x : R.vertices[max_id].neighbor) {
			for (auto y : L.vertices[x - 1].neighbor) {
				if (y > left + 31) break;
				if (y >= left) {
					bit_num[y - left]++;
				}
			}
		}

		//找到1最少的顶点
		unsigned min_bit = 0xFFFFFFFF, min_bit_id = 0xFFFFFFFF;
		for (int i = 0; i < 32; i++) {
			if (remained_vertex.find(i) != remained_vertex.end()) continue;
			if (bit_num[i] < min_bit) {
				min_bit = bit_num[i];
				min_bit_id = i + left;
			}
		}

		if (min_bit_id == 0xFFFFFFFF) continue;
		if (min_bit >= max_nei) continue;

		//交换min_bit_id 和 min_nei_id
		reorder_dic[max_nei_id] = min_bit_id;
		reorder_dic[min_bit_id] = max_nei_id;

		//更新邻居列表
		for (auto x : R.vertices[min_bit_id].neighbor) {
			int res_id = BinarySearch(L.vertices[x - 1].neighbor, L.vertices[x - 1].neighbor.size(), min_bit_id + 1);
			L.vertices[x - 1].neighbor[res_id] = max_nei_id + 1;
			sort(L.vertices[x - 1].neighbor.begin(), L.vertices[x - 1].neighbor.end());
		}

		for (auto x : R.vertices[max_nei_id].neighbor) {
			int res_id = BinarySearch(L.vertices[x - 1].neighbor, L.vertices[x - 1].neighbor.size(), max_nei_id + 1);
			L.vertices[x - 1].neighbor[res_id] = min_bit_id + 1;
			sort(L.vertices[x - 1].neighbor.begin(), L.vertices[x - 1].neighbor.end());
		}

		// 交换两节点的位置
		Vertex tmp_ver = R.vertices[max_nei_id];
		R.vertices[max_nei_id] = R.vertices[min_bit_id];
		R.vertices[min_bit_id] = tmp_ver;

		remained_vertex.erase(max_nei_id);
		//更新vertex_Q
		for (auto x : remained_vertex) {
			vertex_Q[x] = calQofVertex(L, x + 1, R);
		}

		if (remained_vertex.size() % 100 == 0) {
			printf("%ld\n", remained_vertex.size());
		}
	}
}

void calAllQ(Graph& L){
	unsigned res = 0;
	for(int i = 0; i < L.vertex_num_after_trim; i++){
		unsigned off = 0;
		for(auto x : L.vertices[i].neighbor){
			if((x - 1) / 32 > off){
				res++;
				off = (x - 1) / 32;
			}
		}
		res++;
	}
	printf("There are %d\n", res);
}

void printMatrix(Graph& L, unsigned R_num, char* path){
	FILE *fp = fopen(path, "w");
	for(int i = 0; i < L.vertex_num_after_trim; i++){
		unsigned pointer = 1, counter = 0;
		for(auto x : L.vertices[i].neighbor){
			while(pointer < x){
				fprintf(fp, "0 ");
				pointer++;
				counter++;
				if(counter % 32 == 0) fprintf(fp, "|");
			}
			fprintf(fp, "1 ");
			pointer++;
			counter++;
			if(counter % 32 == 0) fprintf(fp, "|");
		}
		while(pointer <= R_num){
			fprintf(fp, "0 ");
			pointer++;
			counter++;
			if(counter % 32 == 0) fprintf(fp, "|");
		}
		fprintf(fp, "\n");
	}
}

void calNumOfOne(Graph& L){
	unsigned res[33] = {0};
	for(int i = 0; i < L.vertex_num_after_trim; i++){
		unsigned off = 0, counter = 0;
		for(auto x : L.vertices[i].neighbor){
			if((x - 1) / 32 > off){
				res[counter]++;
				counter = 1;
				off = (x - 1) / 32;
			}
			else{
				counter++;
			}
		}
		res[counter]++;
	}
	for(int i = 0; i < 32; i++){
		printf("%d ",res[i]);
	}
	printf("\n");
}

unsigned matrix[100000][10000] = { 0 };
// std::vector<std::vector<unsigned>> matrix(80000);

void convertMatrix(Graph& L, unsigned R_num) {
	unsigned row = 0, col = 0;
	for (int i = 0; i < L.vertex_num_after_trim; i++) {
		unsigned pointer = 1, counter = 0;
		col = 0;
		for (auto x : L.vertices[i].neighbor) {
			while (pointer < x) {
				pointer++;
				counter++;
				if (counter % 32 == 0) {
					col++;
					counter = 0;
				}
			}
			matrix[row][col] |= (1 << (counter % 32));
			pointer++;
			counter++;
			if (counter % 32 == 0) {
				counter = 0;
				col++;
			}
		}
		while (pointer <= R_num) {
			pointer++;
			counter++;
			if (counter % 32 == 0) {
				col++;
				counter = 0;
			}
		}
		row++;
	}
}

unsigned CalNumOfOneInBlock(unsigned* BitsArray, unsigned num) {
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

void calNotZeroofMatrix(unsigned l, unsigned r) {
	unsigned res = 0;
	for (int i = 0; i < l; i++) {
		for (int j = 0; j < r; j++) {
			if (matrix[i][j] != 0) {
				res++;
			}
		}
	}
	printf("Not zero in matrix: %d\n", res);
}

unsigned calProfit(unsigned to_exchange, unsigned this_id, unsigned exchanged_one_block_num, unsigned this_one_block_num, Graph& R) {
	// 这一侧，0-块变为1-块
	unsigned zero_to_one_num = 0;
	for (auto x : R.vertices[to_exchange].neighbor) {
		if (matrix[x - 1][this_id / 32] == 0) {
			zero_to_one_num++;
		}
	}
	// 另一侧，0-块变为1-块
	for (auto x : R.vertices[this_id].neighbor) {
		if (matrix[x - 1][to_exchange / 32] == 0) {
			zero_to_one_num++;
		}
	}
	//这一侧，2-块变为1-块
	unsigned two_to_one_num = 0;
	for (auto x : R.vertices[this_id].neighbor) {
		//判断是不是2-块
		if (CalNumOfOneInBlock(&matrix[x - 1][this_id / 32], 1) == 2) {
			//判断移过来的对应位置是不是1
			int res_id = BinarySearchNew(R.vertices[to_exchange].neighbor, R.vertices[to_exchange].neighbor.size(), x);
			if (res_id == -1) two_to_one_num++;
		}
	}

	//另一侧，2-块变为1-块
	for (auto x : R.vertices[to_exchange].neighbor) {
		//判断是不是2-块
		if (CalNumOfOneInBlock(&matrix[x - 1][to_exchange / 32], 1) == 2) {
			//判断移过来的对应位置是不是1
			int res_id = BinarySearchNew(R.vertices[this_id].neighbor, R.vertices[this_id].neighbor.size(), x);
			if (res_id == -1) two_to_one_num++;
		}
	}

	//计算profit
	if (exchanged_one_block_num + this_one_block_num > zero_to_one_num + two_to_one_num) {
		return exchanged_one_block_num + this_one_block_num - zero_to_one_num - two_to_one_num;
	}
	else {
		return 0;
	}
}

unsigned calProfitNew(unsigned to_exchange, unsigned this_id, unsigned exchanged_one_block_num, unsigned this_one_block_num, Graph& R) {
	/*if (this_id == 4483 && to_exchange == 4760) {
		printf("A\n");
	}*/

	// 这一侧，0-块变为1-块
	unsigned zero_to_one_num = 0;
	unsigned one_to_zero_num = 0;
	for (auto x : R.vertices[to_exchange].neighbor) {
		if (matrix[x - 1][this_id / 32] == 0) {
			zero_to_one_num++;
		}
		// 另一侧，1-块变为0-块
		if (CalNumOfOneInBlock(&matrix[x - 1][to_exchange / 32], 1) > 1) continue;
		// int res_id = BinarySearchNew(R.vertices[this_id].neighbor, R.vertices[this_id].neighbor.size(), x);
		// if (res_id == -1) {
		if((matrix[x - 1][this_id / 32] & (0x00000001 << (this_id % 32))) == 0){
			one_to_zero_num++;
		}
	}
	// 另一侧，0-块变为1-块
	for (auto x : R.vertices[this_id].neighbor) {
		if (matrix[x - 1][to_exchange / 32] == 0) {
			zero_to_one_num++;
		}
		// 这一侧，1-块变为0-块
		if (CalNumOfOneInBlock(&matrix[x - 1][this_id / 32], 1) > 1) continue;
		// int res_id = BinarySearchNew(R.vertices[to_exchange].neighbor, R.vertices[to_exchange].neighbor.size(), x);
		// if (res_id == -1) {
		if((matrix[x - 1][to_exchange / 32] & (0x00000001 << (to_exchange % 32))) == 0){
			one_to_zero_num++;
		}
	}

	if (one_to_zero_num > zero_to_one_num) {
		return one_to_zero_num - zero_to_one_num;
	}
	else {
		return 0;
	}
}

void updateOneBlock(unsigned to_exchange, unsigned this_id, Graph& R, std::set<unsigned>& one_block_set, std::map<unsigned, unsigned>& one_block_dic) {
	// 这一侧，0-块变为1-块
	for (auto x : R.vertices[to_exchange].neighbor) {
		if (matrix[x - 1][this_id / 32] == 0) {
			one_block_set.insert(this_id);
			if (one_block_dic.find(this_id) != one_block_dic.end()) {
				one_block_dic[this_id]++;
			}
			else {
				one_block_dic[this_id] = 1;
			}
		}
	}
	// 另一侧，0-块变为1-块
	for (auto x : R.vertices[this_id].neighbor) {
		if (matrix[x - 1][to_exchange / 32] == 0) {
			one_block_set.insert(to_exchange);
			if (one_block_dic.find(to_exchange) != one_block_dic.end()) {
				one_block_dic[to_exchange]++;
			}
			else {
				one_block_dic[to_exchange] = 1;
			}
		}
	}

	//这一侧，2-块变为1-块
	for (auto x : R.vertices[this_id].neighbor) {
		//判断是不是2-块
		if (CalNumOfOneInBlock(&matrix[x - 1][this_id / 32], 1) == 2) {
			//判断移过来的对应位置是不是1
			// int res_id = BinarySearchNew(R.vertices[to_exchange].neighbor, R.vertices[to_exchange].neighbor.size(), x);
			// if (res_id == -1) {
			if((matrix[x - 1][to_exchange / 32] & (0x00000001 << (to_exchange % 32))) == 0){
				unsigned id = 0, tmp_item = matrix[x - 1][this_id / 32];
				while (tmp_item > 0) {
					if ((tmp_item & 1) == 1) {
						if (id != this_id % 32) break;
					}
					tmp_item = tmp_item >> 1;
					id++;
				}
				unsigned cor_id = (this_id / 32) * 32 + id;
				one_block_set.insert(cor_id);
				if (one_block_dic.find(cor_id) != one_block_dic.end()) {
					one_block_dic[cor_id]++;
				}
				else {
					one_block_dic[cor_id] = 1;
				}
			}
		}
	}

	//另一侧，2-块变为1-块
	for (auto x : R.vertices[to_exchange].neighbor) {
		//判断是不是2-块
		if (CalNumOfOneInBlock(&matrix[x - 1][to_exchange / 32], 1) == 2) {
			//判断移过来的对应位置是不是1
			// int res_id = BinarySearchNew(R.vertices[this_id].neighbor, R.vertices[this_id].neighbor.size(), x);
			// if (res_id == -1) {
			if((matrix[x - 1][this_id / 32] & (0x00000001 << (this_id % 32))) == 0){
				unsigned id = 0, tmp_item = matrix[x - 1][to_exchange / 32];
				while (tmp_item > 0) {
					if ((tmp_item & 1) == 1) {
						if (id != to_exchange % 32) break;
					}
					tmp_item = tmp_item >> 1;
					id++;
				}
				unsigned cor_id = (to_exchange / 32) * 32 + id;
				one_block_set.insert(cor_id);
				if (one_block_dic.find(cor_id) != one_block_dic.end()) {
					one_block_dic[cor_id]++;
				}
				else {
					one_block_dic[cor_id] = 1;
				}
			}
		}
	}

	// 这一侧，1-块变为0-块
	for (auto x : R.vertices[this_id].neighbor) {
		if (CalNumOfOneInBlock(&matrix[x - 1][this_id / 32], 1) > 1) continue;
		// int res_id = BinarySearchNew(R.vertices[to_exchange].neighbor, R.vertices[to_exchange].neighbor.size(), x);
		// if (res_id == -1) {
		if((matrix[x - 1][to_exchange / 32] & (0x00000001 << (to_exchange % 32))) == 0){
			if (one_block_dic[this_id] > 0) one_block_dic[this_id]--;
			else {
				one_block_dic.erase(this_id);
				one_block_set.erase(this_id);
			}
		}
	}

	

	// 另一侧，1-块变为0-块
	for (auto x : R.vertices[to_exchange].neighbor) {
		if (CalNumOfOneInBlock(&matrix[x - 1][to_exchange / 32], 1) > 1) continue;
		// int res_id = BinarySearchNew(R.vertices[this_id].neighbor, R.vertices[this_id].neighbor.size(), x);
		// if (res_id == -1) {
		if((matrix[x - 1][this_id / 32] & (0x00000001 << (this_id % 32))) == 0){
			if (one_block_dic[to_exchange] > 0) one_block_dic[to_exchange]--;
			else {
				one_block_dic.erase(to_exchange);
				one_block_set.erase(to_exchange);
			}
		}
	}
}


typedef struct nodewithnei{
	unsigned id;
	unsigned nei_num;
} NWN;

struct cmpMatrix{
	bool operator() (NWN a, NWN b){
		return a.nei_num > b.nei_num;
	}
};

void reorderByMatrix(Graph& L, Graph& R) {
	clock_t start;
	unsigned long long duration = 0;

	std::set<unsigned> one_block_vec;
	std::map<unsigned, unsigned> one_block_dic;
	std::set<unsigned> reordered;
	// unsigned old_size = 0xFFFFFFFF;
	unsigned count_all = 0;
	for(int i = 0; i < L.vertex_num_after_trim; i++){
		for(int j = 0; j < (R.vertex_num_after_trim / 32 + 1); j++){
			matrix[i][j] = 0;
		}
	}

	//首先转换成bitmap矩阵
	convertMatrix(L, R.vertex_num_after_trim);
	calNotZeroofMatrix(L.vertex_num_after_trim, R.vertex_num_after_trim / 32 + 1);

	//遍历矩阵，收集1-块节点
	for (int i = 0; i < L.vertex_num_after_trim; i++) {
		for (unsigned j = 0; j < (R.vertex_num_after_trim / 32 + 1); j++) {
			if (CalNumOfOneInBlock(&matrix[i][j], 1) == 1) {
				unsigned id = 0, tmp_item = matrix[i][j];
				while (tmp_item > 0) {
					tmp_item = tmp_item >> 1;
					id++;
				}
				if(j * 32 + id - 1 < R.vertex_num_after_trim) one_block_vec.insert((j * 32 + id - 1));
			}
		}
	}

	while (!one_block_vec.empty() && count_all < 500) {
		count_all++;
		printf("%d\n",count_all);
		// old_size = one_block_vec.size();
		//寻找1-块节点用于交换
		//step1: 收集节点1-块的数量 1/11
		
		for (auto x : one_block_vec) {
			unsigned of = x;
			if (one_block_dic.find(of) != one_block_dic.end()) {
				one_block_dic[of]++;
			}
			else {
				one_block_dic[of] = 1;
			}
		}
		
		// // step2: 找最多数量的节点 0.04/11
		unsigned max_num = 0, max_id = 0;
		for (auto x : one_block_dic) {
			// if(reordered.find(x.first) != reordered.end()) continue;
			if (x.second >= max_num) {
				max_num = x.second;
				max_id = x.first;
			}
		}
		// unsigned tmp_cnt = 0, rand_id = rand() % one_block_vec.size();
		// for(auto x : one_block_vec){
		// 	tmp_cnt++;
		// 	if(tmp_cnt == rand_id){
		// 		max_id = x;
		// 		max_num = one_block_dic[max_id];
		// 	}
		// }
		// printf("End max-id, %d\n", max_id);

		one_block_vec.erase(max_id);
		reordered.insert(max_id);
		one_block_dic[max_id] = 0;

		//收集该顶点的top-k小邻居的顶点 4/11
		// 3.94/11
		start = clock();
		std::vector<unsigned> min_nei_vertices;
		std::priority_queue<NWN, std::vector<NWN>, cmpMatrix> min_queue; 
		for (unsigned i = 0; i < R.vertex_num_after_trim; i++) {
			if (i / 32 == max_id / 32) continue;
			unsigned com_nei = InterSectionBinary(R.vertices[max_id].neighbor, R.vertices[i].neighbor);
			NWN tmp_stru;
			tmp_stru.id = i;
			tmp_stru.nei_num = com_nei;
			min_queue.push(tmp_stru);
		}
		duration += (clock() - start);
		// printf("End queue\n");
		
		// 0.75/11
		unsigned min_nei = min_queue.top().nei_num, top_k = 0;
		while(!min_queue.empty()){
			if(min_nei == min_queue.top().nei_num){
				min_nei_vertices.push_back(min_queue.top().id);
				min_queue.pop();
			}
			else if(min_nei < min_queue.top().nei_num){
				if(top_k < 0){
					min_nei = min_queue.top().nei_num;
					min_nei_vertices.push_back(min_queue.top().id);
					min_queue.pop();
					top_k++;
				}
				else{
					break;
				}
			}
		}
		
		// printf("End top-k\n");
		// printf("%d,", min_queue.top().nei_num);
		// unsigned min_nei = 0xFFFFFFFF;
		// for (unsigned i = 0; i < R.vertex_num_after_trim; i++) {
		// 	if (i / 32 == max_id / 32) continue;
		// 	unsigned com_nei = InterSectionBinary(R.vertices[max_id].neighbor, R.vertices[i].neighbor);
		// 	if (com_nei < min_nei) {
		// 		min_nei_vertices.clear();
		// 		min_nei = com_nei;
		// 		min_nei_vertices.push_back(i);
		// 	}
		// 	else if (com_nei == min_nei) {
		// 		min_nei_vertices.push_back(i);
		// 	}
		// }

		// 4.79/11
		//遍历top-k小邻居顶点，寻找使得收益最大的顶点并交换 4/11
		unsigned max_profit = 0, max_profit_id = 0;
		for (auto x : min_nei_vertices) {
			unsigned profit = calProfitNew(x, max_id, max_num, one_block_dic[x], R);
			if (profit > max_profit) {
				max_profit = profit;
				max_profit_id = x;
			}
		}
		if (max_profit == 0) continue;
		
		one_block_vec.erase(max_profit_id);
		if (one_block_dic.find(max_profit_id) != one_block_dic.end()) {
			one_block_dic[max_profit_id] = 0;
		}
		
		// printf("%d, %d\n", max_id, max_profit_id);
		//找到后，做交换 0.59/11
		//step1: 更新matrix
		unsigned pointer_max_id = 0, pointer_max_profit_id = 0;
		unsigned max_id_val = R.vertices[max_id].neighbor[pointer_max_id], max_profit_val = R.vertices[max_profit_id].neighbor[pointer_max_profit_id];
		for (auto x : R.vertices[max_id].neighbor) {
			matrix[x - 1][max_id / 32] &= (~(1 << (max_id % 32)));
		}
		for (auto x : R.vertices[max_profit_id].neighbor) {
			matrix[x - 1][max_profit_id / 32] &= (~(1 << (max_profit_id % 32)));
		}
		for (auto x : R.vertices[max_id].neighbor) {
			matrix[x - 1][max_profit_id / 32] |= (1 << (max_profit_id % 32));
		}
		for (auto x : R.vertices[max_profit_id].neighbor) {
			matrix[x - 1][max_id / 32] |= (1 << (max_id % 32));
		}

		// printf("%d %d\n", max_id, max_profit_id);
		//step2: 更新邻居列表
		for (auto x : R.vertices[max_id].neighbor) {
			int res_id = BinarySearch(L.vertices[x - 1].neighbor, L.vertices[x - 1].neighbor.size(), max_id + 1);
			// if(res_id == -1) {
			// 	printf("A %d\n", x);
			// }
			// 	for(auto y : L.vertices[x - 1].neighbor) printf("%d, ", y);
			// 	printf("\n");
			// }
			L.vertices[x - 1].neighbor[res_id] = max_profit_id + 1;
			sort(L.vertices[x - 1].neighbor.begin(), L.vertices[x - 1].neighbor.end());
		}

		for (auto x : R.vertices[max_profit_id].neighbor) {
			int res_id = BinarySearch(L.vertices[x - 1].neighbor, L.vertices[x - 1].neighbor.size(), max_profit_id + 1);
			L.vertices[x - 1].neighbor[res_id] = max_id + 1;
			sort(L.vertices[x - 1].neighbor.begin(), L.vertices[x - 1].neighbor.end());
		}

		//step3: 交换两节点的位置
		Vertex tmp_ver = R.vertices[max_id];
		R.vertices[max_id] = R.vertices[max_profit_id];
		R.vertices[max_profit_id] = tmp_ver;
		
		//step4: 更新1-块节点及其数量
		updateOneBlock(max_profit_id, max_id, R, one_block_vec, one_block_dic);
		
		printf("Remained: %ld, %u, %u\n", one_block_vec.size(), count_all, max_profit);
		//calAllQ(L);
		// calNotZeroofMatrix(L.vertex_num_after_trim, R.vertex_num_after_trim / 32 + 1);
		
	}
	// printf("End\n");
	printf("%.2f s\n", (float)duration / CLOCKS_PER_SEC);
}

void reorderByMatrixNew(Graph& L, Graph& R) {
	// clock_t start;
	// unsigned long long duration = 0;
	omp_set_num_threads(6);
	std::set<unsigned> one_block_vec;
	std::map<unsigned, unsigned> one_block_dic;
	std::set<unsigned> reordered;
	// unsigned old_size = 0xFFFFFFFF;
	unsigned count_all = 0;
	for(int i = 0; i < L.vertex_num_after_trim; i++){
		for(int j = 0; j < (R.vertex_num_after_trim / 32 + 1); j++){
			matrix[i][j] = 0;
		}
	}
	
	//首先转换成bitmap矩阵
	convertMatrix(L, R.vertex_num_after_trim);
	calNotZeroofMatrix(L.vertex_num_after_trim, R.vertex_num_after_trim / 32 + 1);
	//遍历矩阵，收集1-块节点
	for (int i = 0; i < L.vertex_num_after_trim; i++) {
		for (unsigned j = 0; j < (R.vertex_num_after_trim / 32 + 1); j++) {
			if (CalNumOfOneInBlock(&matrix[i][j], 1) == 1) {
				unsigned id = 0, tmp_item = matrix[i][j];
				while (tmp_item > 0) {
					tmp_item = tmp_item >> 1;
					id++;
				}
				if(j * 32 + id - 1 < R.vertex_num_after_trim) one_block_vec.insert((j * 32 + id - 1));
			}
		}
	}
	
	//寻找1-块节点用于交换
	//step1: 收集节点1-块的数量 1/11
	
	for (auto x : one_block_vec) {
		unsigned of = x;
		if (one_block_dic.find(of) != one_block_dic.end()) {
			one_block_dic[of]++;
		}
		else {
			one_block_dic[of] = 1;
		}
	}
	
	while (!one_block_vec.empty() && count_all < 6000) {
		count_all++;
		// printf("%d\n",count_all);
		// old_size = one_block_vec.size();
		
		// start = clock();
		// // step2: 找最多数量的节点 0.15+0.07/0.8
		auto min_map = std::min_element(one_block_dic.begin(), one_block_dic.end(), [](const auto&a, const auto& b){
			return a.second >= b.second;
		});
		// duration += (clock() - start);

		unsigned max_id = min_map->first, max_num = min_map->second;
		one_block_vec.erase(max_id);
		reordered.insert(max_id);
		one_block_dic[max_id] = 0;
		
		//收集该顶点的top-k小邻居的顶点 0.07/0.8
		std::vector<unsigned> min_nei_vertices;
		unsigned* list2hop = new unsigned[R.vertex_num_after_trim]();
		int i_neighbor_num = R.vertices[max_id].neighbor.size();
		for (int j = 0; j < i_neighbor_num; j++) {
			unsigned j_vertex = R.vertices[max_id].neighbor[j];
			int j_neighbor_num = L.vertices[j_vertex - 1].neighbor.size();
			for (int k = 0; k < j_neighbor_num; k++) {
				list2hop[L.vertices[j_vertex - 1].neighbor[k] - 1]++;
			}
		}
		unsigned min_ele = 1000000, min_nei = 1000000;
		for(unsigned i = 0; i < R.vertex_num_after_trim; i++){
			if(list2hop[i] < min_ele) min_ele = list2hop[i];
			if(R.vertices[i].neighbor.size() < min_nei) min_nei = R.vertices[i].neighbor.size();
		}
		for(unsigned i = 0; i < R.vertex_num_after_trim; i++){
			if(R.vertices[i].neighbor.size() == min_nei && list2hop[i] == min_ele) min_nei_vertices.push_back(i);
		}
		delete[] list2hop;
		
		// 0.23 / 0.8
		//遍历top-k小邻居顶点，寻找使得收益最大的顶点并交换
		unsigned max_profit = 0, max_profit_id = 0;
		std::vector<unsigned> profit_arr(min_nei_vertices.size());
		#pragma omp parallel for
		for (unsigned i = 0; i < min_nei_vertices.size(); i++) {
			unsigned profit = calProfitNew(min_nei_vertices[i], max_id, max_num, one_block_dic[min_nei_vertices[i]], R);
			profit_arr[i] = profit;
		}
		for(unsigned i = 0; i < min_nei_vertices.size(); i++){
			if(profit_arr[i] > max_profit){
				max_profit = profit_arr[i];
				max_profit_id = min_nei_vertices[i];
			}
		}
		
		// for (auto x : min_nei_vertices) {
		// 	unsigned profit = calProfitNew(x, max_id, max_num, one_block_dic[x], R);
		// 	if (profit > max_profit) {
		// 		max_profit = profit;
		// 		max_profit_id = x;
		// 	}
		// }

		if (max_profit == 0) continue;
		
		one_block_vec.erase(max_profit_id);
		if (one_block_dic.find(max_profit_id) != one_block_dic.end()) {
			one_block_dic[max_profit_id] = 0;
		}
		
		//找到后，做交换 0.14 / 0.8
		//step1: 更新matrix
		unsigned pointer_max_id = 0, pointer_max_profit_id = 0;
		unsigned max_id_val = R.vertices[max_id].neighbor[pointer_max_id], max_profit_val = R.vertices[max_profit_id].neighbor[pointer_max_profit_id];
		#pragma omp parallel for
		// for (auto x : R.vertices[max_id].neighbor) {
		for(int i = 0; i < R.vertices[max_id].neighbor.size(); i++){
			// 更新matrix
			auto x = R.vertices[max_id].neighbor[i];
			matrix[x - 1][max_id / 32] &= (~(1 << (max_id % 32)));
			matrix[x - 1][max_profit_id / 32] |= (1 << (max_profit_id % 32));
			// 更新邻居列表
			int res_id = BinarySearch(L.vertices[x - 1].neighbor, L.vertices[x - 1].neighbor.size(), max_id + 1);
			L.vertices[x - 1].neighbor[res_id] = max_profit_id + 1;
			sort(L.vertices[x - 1].neighbor.begin(), L.vertices[x - 1].neighbor.end());
		}
		#pragma omp parallel for
		// for (auto x : R.vertices[max_profit_id].neighbor) {
		for(int i = 0; i < R.vertices[max_profit_id].neighbor.size(); i++){
			auto x = R.vertices[max_profit_id].neighbor[i];
			matrix[x - 1][max_profit_id / 32] &= (~(1 << (max_profit_id % 32)));
			matrix[x - 1][max_id / 32] |= (1 << (max_id % 32));
			// 更新邻居列表
			int res_id = BinarySearch(L.vertices[x - 1].neighbor, L.vertices[x - 1].neighbor.size(), max_profit_id + 1);
			L.vertices[x - 1].neighbor[res_id] = max_id + 1;
			sort(L.vertices[x - 1].neighbor.begin(), L.vertices[x - 1].neighbor.end());
		}
		
		//step3: 交换两节点的位置
		Vertex tmp_ver = R.vertices[max_id];
		R.vertices[max_id] = R.vertices[max_profit_id];
		R.vertices[max_profit_id] = tmp_ver;
		
		//step4: 更新1-块节点及其数量
		updateOneBlock(max_profit_id, max_id, R, one_block_vec, one_block_dic);
		// updateOneBlockNew(max_profit_id, max_id, R, one_block_vec, one_block_dic, is_one_block);
		
		// printf("Remained: %ld, %u, %u\n", one_block_vec.size(), count_all, max_profit);

		//calAllQ(L);
		// calNotZeroofMatrix(L.vertex_num_after_trim, R.vertex_num_after_trim / 32 + 1);
		
	}
	// printf("End\n");
	
	// printf("%.2f s\n", (float)duration / CLOCKS_PER_SEC);
}

void reorderByMatrixOld(Graph& L, Graph& R) {
	std::set<unsigned> one_block_vec;
	std::vector<unsigned> one_block_vec_all;
	std::map<unsigned, unsigned> one_block_dic;
	std::set<unsigned> reordered;
	// for (int i = 0; i < 80000; i++) {
	// 	matrix[i].resize(20000);
	// }
	// unsigned old_size = 0xFFFFFFFF;
	unsigned count_all = 0;
	for(int i = 0; i < L.vertex_num_after_trim; i++){
		for(int j = 0; j < (R.vertex_num_after_trim / 32 + 1); j++){
			matrix[i][j] = 0;
		}
	}

	//首先转换成bitmap矩阵
	convertMatrix(L, R.vertex_num_after_trim);
	calNotZeroofMatrix(L.vertex_num_after_trim, R.vertex_num_after_trim / 32 + 1);

	//遍历矩阵，收集1-块节点
	for (int i = 0; i < L.vertex_num_after_trim; i++) {
		for (unsigned j = 0; j < (R.vertex_num_after_trim / 32 + 1); j++) {
			if (CalNumOfOneInBlock(&matrix[i][j], 1) == 1) {
				unsigned id = 0, tmp_item = matrix[i][j];
				while (tmp_item > 0) {
					tmp_item = tmp_item >> 1;
					id++;
				}
				if(j * 32 + id - 1 < R.vertex_num_after_trim) one_block_vec_all.push_back((j * 32 + id - 1));
			}
		}
	}

	for (auto x : one_block_vec_all) {
		unsigned of = x;
		one_block_vec.insert(x);
		if (one_block_dic.find(of) != one_block_dic.end()) {
			one_block_dic[of]++;
		}
		else {
			one_block_dic[of] = 1;
		}
	}

	while (!one_block_vec.empty() && count_all < 8000) {
		count_all++;
		printf("%d\n",count_all);
		// old_size = one_block_vec.size();
		//寻找1-块节点用于交换
		//step1: 收集节点1-块的数量
		
		// // step2: 找最多数量的节点
		unsigned max_num = 0, max_id = 0;
		for (auto x : one_block_dic) {
			// if(reordered.find(x.first) != reordered.end()) continue;
			if (x.second >= max_num) {
				max_num = x.second;
				max_id = x.first;
			}
		}
		// unsigned tmp_cnt = 0, rand_id = rand() % one_block_vec.size();
		// for(auto x : one_block_vec){
		// 	tmp_cnt++;
		// 	if(tmp_cnt == rand_id){
		// 		max_id = x;
		// 		max_num = one_block_dic[max_id];
		// 	}
		// }
		printf("End max-id, %d\n", max_id);

		one_block_vec.erase(max_id);
		reordered.insert(max_id);
		one_block_dic[max_id] = 0;

		//收集该顶点的top-k小邻居的顶点
		std::vector<unsigned> min_nei_vertices;
		std::priority_queue<NWN, std::vector<NWN>, cmpMatrix> min_queue; 
		for (unsigned i = 0; i < R.vertex_num_after_trim; i++) {
			if (i / 32 == max_id / 32) continue;
			unsigned com_nei = InterSectionBinary(R.vertices[max_id].neighbor, R.vertices[i].neighbor);
			NWN tmp_stru;
			tmp_stru.id = i;
			tmp_stru.nei_num = com_nei;
			min_queue.push(tmp_stru);
		}
		
		printf("End queue\n");

		unsigned min_nei = min_queue.top().nei_num, top_k = 0;
		while(!min_queue.empty()){
			if(min_nei == min_queue.top().nei_num){
				min_nei_vertices.push_back(min_queue.top().id);
				min_queue.pop();
			}
			else if(min_nei < min_queue.top().nei_num){
				if(top_k < 0){
					min_nei = min_queue.top().nei_num;
					min_nei_vertices.push_back(min_queue.top().id);
					min_queue.pop();
					top_k++;
				}
				else{
					break;
				}
			}
		}
		
		printf("End top-k\n");
		// printf("%d,", min_queue.top().nei_num);
		// unsigned min_nei = 0xFFFFFFFF;
		// for (unsigned i = 0; i < R.vertex_num_after_trim; i++) {
		// 	if (i / 32 == max_id / 32) continue;
		// 	unsigned com_nei = InterSectionBinary(R.vertices[max_id].neighbor, R.vertices[i].neighbor);
		// 	if (com_nei < min_nei) {
		// 		min_nei_vertices.clear();
		// 		min_nei = com_nei;
		// 		min_nei_vertices.push_back(i);
		// 	}
		// 	else if (com_nei == min_nei) {
		// 		min_nei_vertices.push_back(i);
		// 	}
		// }
		//遍历top-k小邻居顶点，寻找使得收益最大的顶点并交换
		unsigned max_profit = 0, max_profit_id = 0;
		for (auto x : min_nei_vertices) {
			unsigned profit = calProfitNew(x, max_id, max_num, one_block_dic[x], R);
			if (profit > max_profit) {
				max_profit = profit;
				max_profit_id = x;
			}
		}
		if (max_profit == 0) continue;
		one_block_vec.erase(max_profit_id);
		if (one_block_dic.find(max_profit_id) != one_block_dic.end()) {
			one_block_dic[max_profit_id] = 0;
		}
		printf("%d, %d\n", max_id, max_profit_id);
		//找到后，做交换

		//step1: 更新matrix
		unsigned pointer_max_id = 0, pointer_max_profit_id = 0;
		unsigned max_id_val = R.vertices[max_id].neighbor[pointer_max_id], max_profit_val = R.vertices[max_profit_id].neighbor[pointer_max_profit_id];
		for (auto x : R.vertices[max_id].neighbor) {
			matrix[x - 1][max_id / 32] &= (~(1 << (max_id % 32)));
		}
		for (auto x : R.vertices[max_profit_id].neighbor) {
			matrix[x - 1][max_profit_id / 32] &= (~(1 << (max_profit_id % 32)));
		}
		for (auto x : R.vertices[max_id].neighbor) {
			matrix[x - 1][max_profit_id / 32] |= (1 << (max_profit_id % 32));
		}
		for (auto x : R.vertices[max_profit_id].neighbor) {
			matrix[x - 1][max_id / 32] |= (1 << (max_id % 32));
		}

		//step2: 更新邻居列表
		for (auto x : R.vertices[max_id].neighbor) {
			int res_id = BinarySearch(L.vertices[x - 1].neighbor, L.vertices[x - 1].neighbor.size(), max_id + 1);
			L.vertices[x - 1].neighbor[res_id] = max_profit_id + 1;
			sort(L.vertices[x - 1].neighbor.begin(), L.vertices[x - 1].neighbor.end());
		}

		for (auto x : R.vertices[max_profit_id].neighbor) {
			int res_id = BinarySearch(L.vertices[x - 1].neighbor, L.vertices[x - 1].neighbor.size(), max_profit_id + 1);
			L.vertices[x - 1].neighbor[res_id] = max_id + 1;
			sort(L.vertices[x - 1].neighbor.begin(), L.vertices[x - 1].neighbor.end());
		}

		//step3: 交换两节点的位置
		Vertex tmp_ver = R.vertices[max_id];
		R.vertices[max_id] = R.vertices[max_profit_id];
		R.vertices[max_profit_id] = tmp_ver;
		
		//step4: 更新1-块节点及其数量
		updateOneBlock(max_profit_id, max_id, R, one_block_vec, one_block_dic);
		
		printf("Remained: %ld, %u, %u\n", one_block_vec.size(), count_all, max_profit);
		//calAllQ(L);
		// calNotZeroofMatrix(L.vertex_num_after_trim, R.vertex_num_after_trim / 32 + 1);
	}
	// printf("End\n");
}


int main(int argc, char* argv[]){
    unsigned p = atoi(argv[2]), q = atoi(argv[3]);
    char* path = argv[1];
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
		// calAllDis(graphL, graphR);
        TrimGraphByCoreNew(graphL, graphR, p, q);
        std::cout << "vertexNum:" << graphL.vertex_num_after_trim << "; edgeNum:" << graphL.edge_num_after_trim << std::endl;
        std::cout << "vertexNum:" << graphR.vertex_num_after_trim << "; edgeNum:" << graphR.edge_num_after_trim << std::endl;

		reformatGraph(graphL, graphR);
		std::cout << "End reformat" << std::endl;
		// calAllQ(graphL);
		// calNumOfOne(graphL);
        // reorderVertex(graphL, graphR);
        // std::cout << "End reorder" << std::endl;
        // for (int i = 0; i < graphL.vertex_num; i++) {
        //     for (int j = 0; j < graphL.vertices[i].neighbor.size(); j++) {
        //         graphL.vertices[i].neighbor[j] = graphL.vertices[i].neighbor[j] & 0x7FFFFFFF;
        //     }
        // }
		// reorderVertexNew(graphL, graphR, wind, 0);
		// reorderForBitmap(graphL, graphR);
		reorderVertexNewNeighbor(graphL, graphR);
		
        // FILE* fp = NULL;
        // fp = fopen("reformated_old_8.txt", "w");
        // fprintf(fp, "%d %d %d\n", graphL.vertex_num_after_trim, graphR.vertex_num_after_trim, graphL.edge_num_after_trim);
        for (int i = 0; i < graphL.vertex_num_after_trim; i++) {
            sort(graphL.vertices[i].neighbor.begin(), graphL.vertices[i].neighbor.end());
        }
		
		// calAllQ(graphL);
		// calNumOfOne(graphL);
		auto start = std::chrono::high_resolution_clock::now();
		reorderByMatrixNew(graphL, graphR);
		auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = end - start;
        std::cout << "All time: " << duration.count() << "s" << std::endl;

		for (int i = 0; i < graphL.vertex_num_after_trim; i++) {
            sort(graphL.vertices[i].neighbor.begin(), graphL.vertices[i].neighbor.end());
        }
		// calAllQ(graphL);
		// calNumOfOne(graphL);

        // fclose(fp);

		// Graph graphH;
        // Collect2Hop(graphL, graphR, graphH, q);
		// std::cout << "Construct H End" << std::endl;
        // // std::cout << "vertexNum:" << graphH.vertex_num << "; edgeNum:" << graphH.edge_num << std::endl;
        // edgeDirectingByDegree(graphH);
        // std::cout << "vertexNum:" << graphH.vertex_num << "; edgeNum:" << graphH.edge_num << std::endl;
		// FILE* fp2 = fopen("reformat_test_H.txt", "w");
		// for(int i = 0; i < graphH.vertex_num; i++){
		// 	for (int j = 0; j < graphH.vertices[i].neighbor.size(); j++) {
		// 		fprintf(fp2, "%d %d\n", i, graphH.vertices[i].neighbor[j] - 1);
		// 	}
		// }
		// fclose(fp2);
		// reorderVertexH(graphH, wind, graphL, 0);
		// reorderVertexNew(graphR, graphL, wind, 0);
		// reorderVertexNewNeighbor(graphR, graphL);
		// calAllQ(graphL);
		// FILE* fp1 = NULL;
        // fp1 = fopen("reformat_test.txt", "w");
        // fprintf(fp1, "%d %d %d\n", graphL.vertex_num_after_trim, graphR.vertex_num_after_trim, graphL.edge_num_after_trim);
        // for (int i = 0; i < graphL.vertex_num_after_trim; i++) {
        //     for (int j = 0; j < graphL.vertices[i].neighbor.size(); j++) {
        //         // fprintf(fp1, "%d %d\n", i + 1, graphL.vertices[i].neighbor[j]);
		// 		fprintf(fp1, "%d %d\n", graphL.vertices[i].neighbor[j] + graphL.vertex_num_after_trim - 1, i);
		// 		// fprintf(fp1, "%d %d\n", i, graphL.vertices[i].neighbor[j] + graphL.vertex_num_after_trim - 1);
        //     }
        // }
        // fclose(fp1);
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
		// calAllDis(graphL, graphR);
        TrimGraphByCoreNew(graphL, graphR, q, p);
        std::cout << "vertexNum:" << graphL.vertex_num_after_trim << "; edgeNum:" << graphL.edge_num_after_trim << std::endl;
        std::cout << "vertexNum:" << graphR.vertex_num_after_trim << "; edgeNum:" << graphR.edge_num_after_trim << std::endl;

		reformatGraph(graphL, graphR);
		std::cout << "End reformat" << std::endl;

		calAllQ(graphL);
		// printMatrix(graphL, graphR.vertex_num_after_trim, "matrix_before_reorder.txt");
		// calNumOfOne(graphL);
		// reorderByMatrix(graphL, graphR);
        // reorderVertexNew(graphL, graphR, wind, 0);
        // std::cout << "End reorder" << std::endl;
        // for (int i = 0; i < graphL.vertex_num; i++) {
        //     for (int j = 0; j < graphL.vertices[i].neighbor.size(); j++) {
        //         graphL.vertices[i].neighbor[j] = graphL.vertices[i].neighbor[j] & 0x7FFFFFFF;
        //     }
        // }
		// reorderVertexNew(graphL, graphR, wind, 0);
		reorderVertexNewNeighbor(graphL, graphR);
        for (int i = 0; i < graphL.vertex_num_after_trim; i++) {
            sort(graphL.vertices[i].neighbor.begin(), graphL.vertices[i].neighbor.end());
        }
		// calAllQ(graphL);
		// calNumOfOne(graphL);
		
		auto start = std::chrono::high_resolution_clock::now();
		reorderByMatrixNew(graphL, graphR);
		auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = end - start;
        std::cout << "All time: " << duration.count() << "s" << std::endl;

		for (int i = 0; i < graphL.vertex_num_after_trim; i++) {
            sort(graphL.vertices[i].neighbor.begin(), graphL.vertices[i].neighbor.end());
        }
		// calAllQ(graphL);
		// calNumOfOne(graphL);
        // fclose(fp);

		// Graph graphH;
        // Collect2Hop(graphL, graphR, graphH, p);
		// std::cout << "Construct H End" << std::endl;
        // // std::cout << "vertexNum:" << graphH.vertex_num << "; edgeNum:" << graphH.edge_num << std::endl;
        // edgeDirectingByDegree(graphH);
        // std::cout << "vertexNum:" << graphH.vertex_num << "; edgeNum:" << graphH.edge_num << std::endl;

		// reorderVertexH(graphH, wind, graphL, 0);
		// reorderVertexNew(graphR, graphL, wind, 0);
		// reorderVertexNewNeighbor(graphR, graphL);
		// reorderByMatrix(graphR, graphL);
		// printf("End\n");
		// FILE* fp = NULL;
        // fp = fopen("reformat_test.txt", "w");
        // fprintf(fp, "%d %d %d\n", graphL.vertex_num_after_trim, graphR.vertex_num_after_trim, graphL.edge_num_after_trim);
        // for (int i = 0; i < graphL.vertex_num_after_trim; i++) {
        //     for (int j = 0; j < graphL.vertices[i].neighbor.size(); j++) {
        //         // fprintf(fp, "%d %d\n", i + 1, graphL.vertices[i].neighbor[j]);
		// 		fprintf(fp, "%d %d\n", i, graphL.vertices[i].neighbor[j] + graphL.vertex_num_after_trim - 1);
        //     }
        // }
        // fclose(fp);
    }
    return 0;
}