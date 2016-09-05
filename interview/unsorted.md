// ================================================================= //
�Ķ���¼: 2013.10.04, 2013.10.23 2013.11.12
// ================================================================= //

�������Ѹ����ɱ����99%�ĺ������ݴ��������� {// divide and conqure �Ѿ���ӡ�� pdf.
http://blog.csdn.net/v_july_v/article/details/7382693
}

�� 1 billion IP address 4 byte unsigned int, 2 M ram, find an IP that is not in the 1 billion record{
	==> �� 2^16 int count �� 16 λ���ֵ� IP, ������һ������ 2^16, then ��Ըø������� 16 λ��
}

��һ���ά�����꣬��x,yֵ��ʾ������һ������L��Ҫ�󷵻���������Щ���γɵı߳�ΪL��������{	
	==> O(n^2 log n) or O(n^2) (�� hashtable)
	1. sort on x axis, for the same x, sort on y both ascend-- O(nlg(n)) x1 <= .... <= xn
	2. for each pair (n1,n4) check d(n1,n4) == sqrt(2) L  
	   if yes, solve a linear equation, get the axis of other two nodes (n2,n3) of the sqare with (n1,n4) as diagonal
	   check the existences of (n2,n3) in sorted list   
}

���м��� n ���ڵ㣬�Կ��շ�������ͬһʱ��ͬһ�ڵ�ֻ���ջ�ֻ�ܷ������һ�㷨��ʹÿ���ڵ���Լ������нڵ�����ֵ֮��{// �����Ʊ���
	==> solution 1: ����һ������,ͬʱͨѶ������ԭ�⣩�� ���ڵ� id (0...n) �����ƻ����� 2^i �������գ�2^(i-1) ������ ���� 2^i ������; ���߼����; ����� 0 ͳһ���������� 
	==> solution 2: �� 1 ���ƣ����ǲ��� 0 ͳһ�������� �Ӹ�λ��ʼ���� 0 ����
	==> solution 3: ��ÿ��ֻ��һ���շ����� 0 -> 1; ..., n-1 -> n, Ȼ�� n -> n-1, ..., 1-> 0;
}

Generate secret key from a list of <guess, score>{
	if secret key = (4,2,5,3,1), and the guess = (1,2,3,7,1), then its score = 2, because g2 = s2 and g5 = s5.
	Given a sequence of guesses, and scores for each guess, decide if there exists at least one secret key that generates those exact scores.
	==> solution
		1. vector<int> GenerateGuess( vector<int>& FirstGuess, int score) �� nChoosek; 
		2. void Filter(vector<int> & KeySet, vector<int> Guess, int score);
}

����һ�������������Ե��ֵ䡣���еĵ����ǰ���������ҵ����Ե���ĸ˳������ġ����������ҵ����Ե���ĸ˳��{
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

N servers with different url records, got the top ten most frequent records in total url{// ��������
	limit: can not pass massive records to a central server;
	==> ע��: �ϲ����� server �ϵ� top ten ��һ���� contain global top ten. Ҫ�� threshold
	solution Two rounds: (a) first each top ten, merge to get the total top ten; // �ؼ��� set a threshold
			(b) each server estimate the frequences of these ten, denote their minimum as S, merge records with frequence >= S/N; �˲��������� server �ϲ���Threshold S/(N/2),  
}

�� heap ��ʵ�� stack or queue{
	==> ���� index ��Ϊheap �����ֵ
}