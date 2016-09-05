// ================================================================= //
�Ķ���¼: 2013.10.04 2013.10.23
// ================================================================= //

����һ������ͼ������������ͼ�е�һ�����ָ�룬���ظ��Ƶ�ͼ�ж�Ӧ�Ľ��ָ��{//http://www.itint5.com/oj/#46
	==> Ҳ���� Topological sort, ��β����ʵ�� ���� O(VlgV)
	structure GraphNode{
		int val;
		vector<GraphNode*> Neighbor;
		GraphNode(int v):val(v){};
	};
	==> Solution: ���α�������һ�� DFS �����µ���ɵ�Ķ�Ӧ��ϵ�� map ��ʾ���ڶ��� DFS copy. ���������ʱ����ԭͼ�����ݣ��罫�µ�ָ��ӵ��ɵ� neighbor ��󣬿��Բ��� map, ��ʡ���� map.find() �ļ�����
	==> ע��: (1) ��û����ȫ copy ��ͼʱ����Ҫɾ���ɵ����µ�Ķ�Ӧ��ϵ. (2) ������� node structure �� recursive, ���� iterative
	void DFS_visit(GraphNode *node, map<GraphNode*, GraphNode*>& m){
		GraphNode * newNode = new GraphNode(node->data);// error: copy node ���� newNode
		m[node] = newNode;
		for(int i = 0; i < node->neighbors.size(); ++i){
			if( m.find(node->neighbors[i]) == m.end()){// new node
				DFS_visit(node->neighbors[i],m);// visit new nodes
			}
		}
	}
	
	void DFS_copy(GraphNode *node, map<GraphNode*, GraphNode*>& m){
		for(int i = 0; i < node->neighbors.size(); ++i){
			m[node]->neighbors.push_back(m[node->neighbors[i]]);
			if( m[node->neighbors[i]]->neighbors.empty() ){// new node
				DFS_copy(node->neighbors[i],m);// copy new nodes
			}		
		}
	}
	
	GraphNode *cloneGraph(GraphNode *node) {
		if(node == NULL) return NULL;// error: ���ǽ��б߽��ж�
	  map<GraphNode*, GraphNode*> m;
	  DFS_visit(node,m);
	  DFS_copy(node,m);
	  return m[node];
	}
}

Topological Sort{// ������ dependency ������ 
	visit(i,int& time,map&M){
		time += 1;
		i.d = time;
		i.color = grey;
		for (i,j):
			if j.color == white��
				visit(j,time,M)	
		time += 1;		
		i.f = time;
		M[time] = i;
	}
	void main(){
		time = 0;
		map M;
		for( i in V){// ��������٣���Ϊ�����ܵõ����� forest ��ֻ��һ�� tree
			if i.color == white:
				visit(i,time,M)
		}
	}
}

������ƣ�����ÿ�־������һ�Σ�һ������������־�, input����ÿ��������ľƵ�id��ÿ�����������10��. ����������ľ�{
	==> Maximum matching problem // matching problem, ������ �� --> ��, ������Լ������Ϊ������� ת��Ϊ MaxFlow Problem
}

// ================================================================= //

Node: ��� d-������ d+�� �� d = d- + d+
ͨ·(path) <--> ��·(cycle/ curcuit)

==> ����ͼ��, ���� * 2 = node �Ķ����� ==> ����ͼ��, ��ȵ������Ϊż��
==> Euler ��·: ����ÿ��һ���ҽ�һ���һص��յ㡣 Euler ͼ������ Euler ��·��ͼ(�� Euler ͼû��Ҫ��ص��յ㣬ע�� Euler ͼ��һ�����ڰ� Euler ͼ)
	--> ������ͨͼ�� Euler ͼ <==> ͼ������� <==> ͼ�ı߼��ɷ�Ϊ���ɸ����Ȼ�·
	--> ������ͨͼ�� �� Euler ͼ <==> ͼ�����ҽ���������� <==> ͼ�ı߼��ɷ�Ϊ���ɸ����Ȼ�·
	--> ������ͨͼ�� Euler "��·" <==> ͼ��ǡ�������
	--> ������ͨͼ�� Euler ͼ <==> G ��ÿ����� == ����
	

==> Hamilton ��·: ����ÿ��һ���ҽ�һ��
�й���·����: ������СȨ�صı�ʹ֮��Ϊ Euler ͼ

// ================================================================= //
��: ��ͨ����Ȧ����ͼ
	T is Tree <==> T ��Ȧ�� E = N - 1; <==> T ��ͨ �� E = N - 1 <==> T ��һ�±߼���Ȧ <==> T ȥһ������ͨ <==> T ��������Ψһ path ����

���Ŷ����� (Huffman ���� ) --> ���ż���


Word Ladder, Given two words (start and end), and a dictionary, find the shortest path from start to end{
		1. Only one letter can be changed at a time; 2. Each intermediate word must exist in the dictionary
		==> solution 1: �� Linklist �� case ��ͨ��
		==> solution 2: ֱ���� Dict ����ң��� dict ��Ԫ�ذ� Dijistra "����"  queue 
			* ÿ�α��� Dict �� ���������ַ������� ��Ϊ���Կ��ܵ�neighbor ��� 25*L, ���������� dict ����һ�Ρ� ���� degree �н��ͼ����ÿ������ȱ������� Edge set ��
		==> Word Ladder II: ������п��ܵ����·��: ��ΪҪ����֮ǰ����Ԫ�� queue ��� vector, ������ int c, NextPos ����¼λ�ơ� �� unordered_map<string,int> CurrLev, NextLev; ��¼ vector ���λ�á��� vector< set<int> > Pre ����ǰ����� (�� CurrLev ���Ѽ�)
}

�����е� shortest paths{
	==> solution: �� Dijistra �㷨����ÿ���ڵ��ڸ�����̾���ʱ�����ж���·������̾��룬����Ҫ������Щ preNodes
}

���ŴӴ�С���ɾ��Σ�����ľ��α����ܹ��ڸ���ƽ������ȫ��������ľ����У�ÿ�������������һ�����Σ������Űڷŷ���ʹ����ռ�������С{
	==> ������������DAG�������ٵ�·���������е㣬ת���ɶ���ͼƥ��
}

����ͼƥ��{
==> ����ͼG�Ƕ���ͼ: ��� G �ж�����Էָ������ A, B��ʹ�ö� G �����бߵĶ��㶼�ֱ��� A,B �С�
==> ����ͼƥ�䣺�Զ���ͼ G��һ����ͼM��M�ı߼��е����������߶���������ͬһ�����㣬���M��һ��ƥ�䡣
==> ���ƥ�䣺ͼ�а�����������ƥ���Ϊͼ�����ƥ�䡣
==> ����ƥ�䣺������е㶼��ƥ����ϣ����������ƥ��������ƥ�䡣
==> �����ͼ���ƥ������������(Maximal Flow)�����������㷨(Hungarian Algorithm)
==> ��С�㸲��: ���ٵĵ����������еı�
==> ����ͼ�е����ƥ���� == ���ͼ�е���С�㸲���� (Konig����) 
==> ·������: ��һ������ͼ��,��һЩ·����ʹ֮�κ�һ����������ֻ��һ��·����֮���� (ÿ��·������һ������ͨ�Ӽ�)
==> ��С·������ == |G| - ���ƥ����
==> ��������ͼ�������������㶼�������Ķ��㼯��
==> ����ͼ�������� == ������-����ͼ���ƥ��
}