// ================================================================= //
阅读记录: 2013.10.04 2013.10.23
// ================================================================= //

复制一个有向图。输入是有向图中的一个结点指针，返回复制的图中对应的结点指针{//http://www.itint5.com/oj/#46
	==> 也可用 Topological sort, 从尾建点实现 不用 O(VlgV)
	structure GraphNode{
		int val;
		vector<GraphNode*> Neighbor;
		GraphNode(int v):val(v){};
	};
	==> Solution: 三次遍历，第一次 DFS 建立新点与旧点的对应关系用 map 表示。第二次 DFS copy. 如果可以暂时更改原图的数据，如将新点指针加到旧点 neighbor 最后，可以不用 map, 且省下了 map.find() 的计算量
	==> 注意: (1) 在没有完全 copy 旧图时，不要删除旧点与新点的对应关系. (2) 最好利用 node structure 用 recursive, 不用 iterative
	void DFS_visit(GraphNode *node, map<GraphNode*, GraphNode*>& m){
		GraphNode * newNode = new GraphNode(node->data);// error: copy node 成了 newNode
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
		if(node == NULL) return NULL;// error: 忘记进行边界判断
	  map<GraphNode*, GraphNode*> m;
	  DFS_visit(node,m);
	  DFS_copy(node,m);
	  return m[node];
	}
}

Topological Sort{// 用于找 dependency 很有用 
	visit(i,int& time,map&M){
		time += 1;
		i.d = time;
		i.color = grey;
		for (i,j):
			if j.color == white：
				visit(j,time,M)	
		time += 1;		
		i.f = time;
		M[time] = i;
	}
	void main(){
		time = 0;
		map M;
		for( i in V){// 这个不能少，因为极可能得到的是 forest 不只是一棵 tree
			if i.color == white:
				visit(i,time,M)
		}
	}
}

朋友买酒，但是每种酒最多卖一次，一个人最多买三种酒, input给了每个人想买的酒的id，每个人最多想买10种. 如何卖出最多的酒{
	==> Maximum matching problem // matching problem, 两层结点 人 --> 酒, 个数的约束新增为其他点边 转化为 MaxFlow Problem
}

// ================================================================= //

Node: 入度 d-，出度 d+， 度 d = d- + d+
通路(path) <--> 回路(cycle/ curcuit)

==> 任意图中, 边数 * 2 = node 的度数和 ==> 任意图中, 奇度点个数必为偶个
==> Euler 回路: 经过每边一次且仅一次且回到终点。 Euler 图，具有 Euler 回路的图(半 Euler 图没有要求回到终点，注意 Euler 图不一定属于半 Euler 图)
	--> 无向连通图是 Euler 图 <==> 图中无奇点 <==> 图的边集可分为若干个初等回路
	--> 无向连通图是 半 Euler 图 <==> 图中有且仅有两个奇点 <==> 图的边集可分为若干个初等回路
	--> 无向连通图有 Euler "道路" <==> 图中恰有两奇点
	--> 有向连通图是 Euler 图 <==> G 中每点入次 == 出次
	

==> Hamilton 回路: 给过每点一次且仅一次
中国邮路问题: 即加最小权重的边使之成为 Euler 图

// ================================================================= //
树: 连通不含圈无向图
	T is Tree <==> T 无圈且 E = N - 1; <==> T 连通 且 E = N - 1 <==> T 加一新边即有圈 <==> T 去一边则不连通 <==> T 任两点有唯一 path 相连

最优二叉树 (Huffman 编码 ) --> 最优检索


Word Ladder, Given two words (start and end), and a dictionary, find the shortest path from start to end{
		1. Only one letter can be changed at a time; 2. Each intermediate word must exist in the dictionary
		==> solution 1: 建 Linklist 大 case 不通过
		==> solution 2: 直接在 Dict 里查找，将 dict 里元素按 Dijistra "移入"  queue 
			* 每次遍历 Dict 比 尝试其他字符更慢。 因为所以可能的neighbor 最多 25*L, 而不用整个 dict 遍历一次。 对于 degree 有界的图，就每点遍历比遍历整个 Edge set 好
		==> Word Ladder II: 输出所有可能的最短路径: 因为要保留之前所有元素 queue 变成 vector, 用两个 int c, NextPos 来记录位移。 用 unordered_map<string,int> CurrLev, NextLev; 记录 vector 里的位置。用 vector< set<int> > Pre 保持前驱结点 (在 CurrLev 里搜集)
}

找所有的 shortest paths{
	==> solution: 用 Dijistra 算法，但每个节点在更新最短距离时，若有多条路径有最短距离，则需要保持这些 preNodes
}

叠放从大到小若干矩形，上面的矩形必须能够在各边平行下完全放入下面的矩形中，每个矩形上至多放一个矩形，求最优摆放方案使得总占地面积最小{
	==> 拓扑排序后产生DAG，求最少的路径覆盖所有点，转化成二分图匹配
}

二分图匹配{
==> 无向图G是二分图: 如果 G 中顶点可以分割成两部 A, B，使得对 G 中所有边的顶点都分别在 A,B 中。
==> 二分图匹配：对二分图 G的一个子图M，M的边集中的任意两条边都不依附于同一个顶点，则称M是一个匹配。
==> 最大匹配：图中包含边数最多的匹配称为图的最大匹配。
==> 完美匹配：如果所有点都在匹配边上，则称这个最大匹配是完美匹配。
==> 求二分图最大匹配可以用最大流(Maximal Flow)或者匈牙利算法(Hungarian Algorithm)
==> 最小点覆盖: 最少的点来覆盖所有的边
==> 二分图中的最大匹配数 == 这个图中的最小点覆盖数 (Konig定理) 
==> 路径覆盖: 在一个有向图中,找一些路经，使之任何一个顶点有且只有一条路径与之关联 (每条路径就是一个弱连通子集)
==> 最小路径覆盖 == |G| - 最大匹配数
==> 独立集：图中任意两个顶点都不相连的顶点集合
==> 二分图最大独立集 == 顶点数-二分图最大匹配
}