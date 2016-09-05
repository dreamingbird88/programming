// ================================================================= //
阅读记录: 2013.10.04, 2013.10.23 2013.11.12
// ================================================================= //

教你如何迅速秒杀掉：99%的海量数据处理面试题 {// divide and conqure 已经打印成 pdf.
http://blog.csdn.net/v_july_v/article/details/7382693
}

给 1 billion IP address 4 byte unsigned int, 2 M ram, find an IP that is not in the 1 billion record{
	==> 用 2^16 int count 上 16 位出现的 IP, 至少有一个少于 2^16, then 针对该个再找下 16 位的
}

有一组二维点坐标，由x,y值表示，给定一个长度L，要求返回所有由这些点形成的边长为L的正方形{	
	==> O(n^2 log n) or O(n^2) (用 hashtable)
	1. sort on x axis, for the same x, sort on y both ascend-- O(nlg(n)) x1 <= .... <= xn
	2. for each pair (n1,n4) check d(n1,n4) == sqrt(2) L  
	   if yes, solve a linear equation, get the axis of other two nodes (n2,n3) of the sqare with (n1,n4) as diagonal
	   check the existences of (n2,n3) in sorted list   
}

并行计算 n 个节点，皆可收发，但是同一时间同一节点只能收或只能发。设计一算法，使每个节点可以计算所有节点所含值之和{// 二进制编码
	==> solution 1: 若可一发多收,同时通讯（不是原题）， 将节点 id (0...n) 二进制化，被 2^i 整除者收，2^(i-1) 整除但 不被 2^i 整除发; 收者计算和; 最后由 0 统一发，所有收 
	==> solution 2: 与 1 相似，但是不是 0 统一发，而是 从高位开始，由 0 返回
	==> solution 3: 若每次只可一对收发，则 0 -> 1; ..., n-1 -> n, 然后 n -> n-1, ..., 1-> 0;
}

Generate secret key from a list of <guess, score>{
	if secret key = (4,2,5,3,1), and the guess = (1,2,3,7,1), then its score = 2, because g2 = s2 and g5 = s5.
	Given a sequence of guesses, and scores for each guess, decide if there exists at least one secret key that generates those exact scores.
	==> solution
		1. vector<int> GenerateGuess( vector<int>& FirstGuess, int score) 用 nChoosek; 
		2. void Filter(vector<int> & KeySet, vector<int> Guess, int score);
}

给你一本其他国家语言的字典。其中的单词是按照这个国家的语言的字母顺序排序的。输出这个国家的语言的字母顺序{
	==> graph + topological sorting
	vector<char> GetRelation(string s1, string s2){
		vector<char> result(2,0);
		for(int i = 0; i < s1.length() && i < s2.length() && s1[i] == s2[i]; ++i);
		if(i < s1.length() && i < s2.length()){
			result[0] = s1[i], result[1] = s2[i];
		}
		return result;		
	}
	
	void visit(Node* n, int &t, map<int,char>& order){		
		t = t + 1;
		n->color = 1;
		for(auto i = n->neighbor.begin(); i != n->neighbor.end(); ++i){
			if(i->color == 0){
				visit(n->neighbor[i],t,order);
			}
		}
		order[t++] = n->c;		
	}
	map<char,Node*> m;
	for(int i = 0; i < Dict.size() - 2; ++i){
		vector<char> temp = GetRelation(Dict[i],Dict[i+1]);
		if(temp[0] != 0){			
			if(m.find(temp[0]) == m.end()) m[temp[0]] = new Node(temp[0]);
			if(m.find(temp[1]) == m.end()) m[temp[1]] = new Node(temp[1]);
			if(m[temp[0]].neighbor.find(m[temp[1]]) == m[temp[0]].neighbor.end()) m[temp[0]].neighbor.insert(m[temp[1]]);
		}
	}
	
	map<int,char> order;
	for(auto i = m.begin(); i != m.end(); ++i){
		visit(i->second,t,order);
	}		
	
	for(auto i = order.begin(); i != order.end(); ++i){
		cout << i->second << " ";
	}
	cout << endl;

}

N servers with different url records, got the top ten most frequent records in total url{// 海量数据
	limit: can not pass massive records to a central server;
	==> 注意: 合并所有 server 上的 top ten 不一定就 contain global top ten. 要有 threshold
	solution Two rounds: (a) first each top ten, merge to get the total top ten; // 关键是 set a threshold
			(b) each server estimate the frequences of these ten, denote their minimum as S, merge records with frequence >= S/N; 此步可以两两 server 合并新Threshold S/(N/2),  
}

用 heap 来实现 stack or queue{
	==> 即以 index 作为heap 排序的值
}