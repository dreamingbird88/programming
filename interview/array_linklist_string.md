Reading records:
 * 2016.08.21 ĩ���
 * 2013.10.04
 * 2013.10.24
 * 2013.11.12

Sorting:
 * �� quick sort ��ȣ�merge sort ������ص��ǣ�����һ���ȶ������򷽷���merge sort һ������������򡣵��������ŷ���Ҳռ����Ҫ��λ����Ϊ���ǻ��ڱȽϵ�ʱ�临�Ӷ�ΪO(n*lg(n))�������㷨��Ψһ�ȶ���������������Ҫ�ȶ�������ʱͨ����ѡ��merge sort��merge sort ��Ҫ������п��Ժܿ�ؽ���������ʣ����������������ʵ���к��ܻ�ӭ��
 * Ҫ�����ռ�Ļ�����һ�����Ƿ���ù̶���Ŀ����(e.g. �����ķ�Χ)���ڶ������Ƿ��������ԭ�пռ� (e.g. ԭ������)
 * sorting for duplicates prefer �� BST + LinkList
 * ���� indirect sort ������ storage and swapping cost, e.g. �� indirect counting sort �ɽ�ʡ�ռ�


Rolling max (or min) of an array (n) with window(w){// ���� index queue ��������Ϣ Q �� w, Q[0] is the index of the maximium element within (k-w,k]
	define a de-queue, where the front is the index of maximum before that index,
	since each element enter/remove from that queue at most twice, the complexity is just O(n)
	
	void maxSlidingWindow(int A[], int n, int w, int B[]) {
		if(n <= 0 || n < w) return;
	  deque<int> Q;
	  for (int i = 0; i < n; ++i) {
	    if(i >= w) B[i-w] = A[Q.front()];
	    while (!Q.empty() && A[i] >= A[Q.back()]) Q.pop_back(); // ���� A[i] �Ĵ�С���Ӻ����, �ؼ��˲�
	    while (!Q.empty() && Q.front() <= i-w) Q.pop_front(); // ���� i �Ĵ�С����ǰ����
	    Q.push_back(i);
	  }
	  B[n-w] = A[Q.front()];
	}
}

maximum subarray sum {//
	==> solution O(n): �ȼ��� cumulative sum, O(n); ���� sum[i] - min_sum[i] O(n)
}

Rolling sum (or mean) of an array (n) with window (w){//two sum ��� w, ����Ϊ sum with length w.
	void RollingSum(int A[],int n, int w, int B[]){
		if(n <= 0 || n < w) return;
		int i = 0;
		while(++i < n) A[i] += A[i-1]; 
		while(int i = w; i < n; ++i)B[i] = A[i] - A[i-w];
}

Rolling median of online stream{
	==> solution: ά�� min-heap and max-heap, ʹ������Ϊ1, (median ��Ϊ����֮root, �� size ��ȣ���Ϊ�� roots ֮ƽ��ֵ) O(nlgn) Amortized O(lg)
	==> Extension 1: ������� window �� median? 
			--> Solution: maintain two binary tree, ����(max heap) �����Ϊ����(min heap) ��һ������one queue ά�ֹ��ڽ��ָ�롣
	==> Extension 2: rolling top kth statistics of online stream 
			* �� top k ��һ������Ϊ����Ԫ�ع������⣬������������� queue ��¼����Ԫ���� tree node ���ָ�� (��Ҫ�� parent ָ�룬��û��ֻ���� linkedlist ��ɾ�� cur node ͨ�� copy �ӽ�����ﵽ ɾ����Ŀ��)
			* �� rolling median Ҳ��һ������Ϊ median ������������ƽ���������� windows �ڵ�Ԫ��
			--> ��������� rolling median ͬ��ά�� two tree, һ���� min tree �� top K Ԫ�أ�һ���� max tree �� windows �ڵ�����Ԫ�� (��Щ��������������)�� ���� queue<node*> ������� index. ������rolling Top K Ԫ�� complexity Ϊ O(nlgk)
	==> Extension 3: ��һ����֪���ȵ��������� �� k th element, Ҫ�� O(n) time, O(k) space;
			--> solution 1: min-heap O(n lgk); solution 2 (��): Garbage collection ����: ά�� 2k-1 ���飬ÿ�� 2k-1 ʱ ����һ�Σ����� top k ��
}

ά������Ԫ�ص� Min-Stack{
==> Min-Stack, with pop(), push() and min() in O(1) // �� rolling min Q
		Solution 1: double the space, Using additional stack to store CurrentMin.
		Solution 2: O(n) computation in pop() if the popped element is the minimum element
		Solution 3: maintain an auxilary min-index stack, Q[end] = current mininum element index; Q[k-1] = the index of the minimum element after Q[k] is removed, ���� Q ���Ԫ�ؿ϶��Ƿ�����
}

ά������Ԫ�ص� Min-Queue{ // with pop(), push() and min() in O(1)
		Solution 1: double the space, Using additional stack to store CurrentMin.
		Solution 2: ��һ�� index queue �� timestamp, ��Ԫ��ʱ���� q.back() > A[i], q.push_back(t); ��Ԫ��ʱ: �� timestampe - q.front() < Queue size ʱ��pop()
}

���� n �������� x, ʹ�� \sum_{n} |v_n - x| ��С��������λ��{
	==> Extension: ��һ��¥��һ����N�㣬Ҫȥÿһ����˷ֱ���A[1],A[2]....A[N]��������ݿ���ͣK�Σ���ͣ����K�����������ߵľ������
	==> Solution: opt_k[j]--����ǰk��ͣ��1��j��֮�� 1-j ���˵����Ž�, cost[i][j]--��i�㵽��j��ͣһ���� [i,j]֮����������Ž�
		opt_{k+1}[i]=min_t{opt_k[t]+cost[t+1,i];} (i<=n) ���� opt_1[k] = c[1][k], �� c[i,j] = median(i,j) �� rolling median ���ֻ�� O(n^2)
		����̬�ݹ�ֻ�� O(k*n*n), ���Թ��� O(k*n^2)
}

������ S = {1,...,n}, �����ȳ� index array A[K], B[K] with  {A[k],B[k]} ��ͬһ����. �� A[K], B[K] �� 1,...,n �ֳɵ������Ӽ�{
	==> ˼·: ��һ�� I[n], ��ʼ��Ϊ -1; I[i] Ϊ Ԫ�� i ��Ӧ�� list ��ָ��(��û���βָ�룬����ϲ�); ÿ��һԪ�� A[k], �� I[A[k]] Ϊ�գ�����ָ��; ���ǿ������� list ��ͬ������ list �ӵ�ԭlist �ϣ��Ҹ����� list �ϸ�Ԫ�ص� I[i];
}

Given an array of numbers, nums, return an array of numbers products, where products[i] is the product of all nums[j], j != i.{
	Input : [1, 2, 3, 4, 5]
	Output: [(2*3*4*5), (1*3*4*5), (1*2*4*5), (1*2*3*5), (1*2*3*4)] = [120, 60, 40, 30, 24]
	You must do this in O(N) without using division. 
	==> solution: ���߼��� Prefix and Suffix
}

Find the Minimum Non-Existing Positive Integer{ // ���� array index ����Ϣ���ж�ȱ��Ԫ�ػ����ظ�Ԫ��
[3,4,-1,1] returns 2 // ע�� 4 �� 1 ����ʱ��1 ����Ҫ��������

Can you do it in O(n) time and O(1) space?
==> Solution: swap n to a[n], then find the first k, a[k] != k;
or Find all duplicates in an array of size N where the values in the array can be from 0-N

int MinNonExistPosInt(int * Array, int Len){
	for(int i = 0; i < Len; ++i)	{
		while(Array[i] > 0 && Array[i] <= Len)		{
			if(Array[i] != Array[Array[i]-1]){
				int temp = Array[Array[i]-1]; //��ȷ: �����������±��Ѿ��ı�
				Array[Array[i]-1] = Array[i];
				Array[i] = temp;
			}else{
				break;
			}
		}
	}
	for(int i = 0; i < Len; ++i)if(i+1 != Array[i])return i+1;
	return Len+1;
}	
}

Find the median of two sorted arrays (can be different sizes){
	==> �������Ϊ�� �� k ��Ԫ�� �ȽϺ�д code.
    double findKthElement(int A[], int m, int B[], int n, int k) {// m + n >= k
        if(m == 0) return B[k-1];
        if(n == 0) return A[k-1];
        if(k == 1) return min(A[0],B[0]);
        int a = min((k-1)/2,m-1);
        int b = min(k-2-a,n-1); // a + b <= k - 2 ����д��ĵط�
        if(A[a] == B[b]) return B[a];
        // if A[a] < B[b];  for A[i], i <= a, at most i+b <= a+b <= k - 2 element <= A[i], therefore eleminate A[i] up to i == a
        return A[a] < B[b]? findKthElement(A+a+1,m-a-1,B,n,k-a-1): findKthElement(A,m,B+b+1,n-b-1,k-b-1);
    }
}

Given two sorted arrays A, B of size m and n respectively. Find the k-th smallest element in the union of A and B{

You can assume that there are no duplicate elements.

==> solution: O(lg(min(m,n,k))), or lg(k) ÿ��ȥ�� k/2 �� (����һ�� array ���ǰ k/2��)�����ַ� 
	int KthElementTwoArray(int * A, int a, int * B, int b,k){
		if(a == 0 && b == 0 || k > a + b) throw error;
		if(a == 0) return b[k-1];
		if(b == 0) return a[k-1];
		int ai = min(k/2,a-1), bi = min(k/2,b-1);
		return a[ai] < b[bi]? KthElementTwoArray(A+ai+1,a-ai-1,B,b,k-ai-1):KthElementTwoArray(A,a,B+bi+1,b-bi-1,k-bi-1);
	}
	
==> Extension:
	(1) Given a N*N Matrix. All rows are sorted, and all columns are sorted. Find the Kth Largest element of the matrix.(Young Matrix?)
			==> solution: �����kthԪ�����ڵķ��Խ��ߣ����ڸ÷��Խ����ڲ��� O(min(k,n))
	(2) what about M arrays? 
			==> solution 1: O(klgk) using k-heap, each element needs lg(k) // too bad
			==> solution 2: ������һ��������ÿ��ֻ��ȥ M/k * 2 ��Ԫ��
}

2D Young �� matrix ���� �Ƿ����ĳ��{
	==> ÿ�αȽ� A[0][n-1]; �ɴ�ȷ�� ɾ�� �л�����
}


��Ϊn������������median�����k������ҪO(n)��complexity{//Find unsorted ����ǰ k ����ֻ��Ҫ O(n);
	==> Solution 1: O(nlgn) �� sort then O(k);
	==> Solution 2: O(n+k) (a) Find median with Quick sort O(n) amortized: �� RANDOMLY select pivot value, �� split array; (b) �ٽ� 2 * k ɨ��һ�鼴��
}

�� partially sorted array A[n], ÿ��Ԫ��ƫ����ȷλ�� at most k, sort ������� O(nlgk) in space{
	==> �� k length min-heap, ÿ����λһ�����ٵ��� heap O(lgk)
}

10.10 �ж� heap �е�kԪ���� x �Ĺ�ϵ, Ҫ��ʱ���� space ���� O(k) {// ???
	==> �� EquCnt, LagCnt���� O(k) 
}

KMP�����ַ��������㷨(Knuth-Morris-Pratt) search a word w in a string s, O(w+s) instead of O(w*s) {
==> Build an array for length of w: I[0] = -1; where I[i] meaning index of the elment in W should be compared when the ith element is not equal. e.g

���� a b a b ���� -1 0 0 1, ��Ӧ�� -1 0 -1 0, ��һ�� 

int * get_KMP_Ind(char*str, int Len){
	if(Len == 0) return 0;
	if(Len == 1) return new int(-1);
	int * I = new int(Len);
	I[0] = -1; 
	int i = 0, j = -1;
	while(i < L-1){// compare str[0:j] with str[i-j:i], �� ++i ���������� i < L-1 ������ i < L
		if(j == -1 || str[i] == str[j]){
			++j, ++i;// shift both to next charactor because of the same or begining
			I[i] = str[i] == str[j]?I[j]:j; // if str[i] == str[j], then I[i] should be I[j], otherwise, I[i] should be j;
		}else{// if current charactors are not the same, shorten str[0:j] to str[0:I[j]] and go to next comparation
			j = I[j]
		}
	}
	return I;
}
int strstr(char* str, in strlen, char*pattern, int plen){
	int * KMP_Ind = get_KMP_Ind(pattern,plen);
	int pi = 0,si = 0;
	while(si < strlen){
		if(KMP_Ind[pi] == -1 || str[si] == pattern[pi]){
			++si,++pi;
			if(pi == plen)break;
		}else{
			pi = KMP_Ind[pi];
		}
	}
	if(KMP_Ind)free(KMP_Ind);
	return pi == plen? si - plen:-1;
}

BM Boyer-Moore �㷨{// �� String Pattern ����룬�� P ����ַ���ʼ�Ƚϡ� strsub ʵ��ʹ�õ�
	http://blog.csdn.net/v_JULY_v/article/details/6545192
	http://www.inf.fh-flensburg.de/lang/algorithmen/pattern/bmen.htm
	http://www.ruanyifeng.com/blog/2013/05/boyer-moore_string_search_algorithm.html
	BM Boyer-Moore �㷨 �� KMP �㷨���ƣ�KMP �����ǰ׺�飬�� BM ����ú�׺�飬�����õݹ飬�Ƚϸ��� O(n)
	BM�㷨������ ���ַ����� ��ú�׺���� �е����λ��, ����һ�� mismatch ʱ������ bad character �� good suffix ��λ��
	1) Bad character heuristics�� һ����ĸ�� index����pattern ��û�У��򳤶�Ϊ Pattern_length, ����Ϊpattern ��Ӻ�ǰ�����λ��
	2) Good suffix heuristics�� 
}

Given a string S, find the longest palindromic substring in S{
	==> Solution 1, O(n^2): check each element as center of possible substring, need to consider "abba", not just "aba", ���� �ո�
	==> Solution 2, O(n^2): DP; p[i,j] = true if i...j is palindromic. Initial p[i,i] = true, p[i,i+1] = true is s[i] == s[i+1]; p[i-1,j+1] = p[i,j] && s[i-1] == s[j+1];
	==> Solution 3, O(n): Let Len[i] be the length of palindromic with center on i; if c is the index of current center and d <= Len[c], then Len[c-d] <= Len[c+d]; when �� c-d Ϊ���ĵ��Ӵ��� �� c Ϊ���ĵĴ���ʱ(�� c-Len[c] < c-d - Len[c-d]) ���� Len[c+d] ��ĩβʱ, �Ⱥų���. ���ȺŲ������������ c+d Ϊ����(ע�� d �� 1 �� Len[c]) 
	==> Solution 4, O(n): Suffix trees. S and S', if the LP Substring center at c, then the center in S' is at n-c+1. check whether (c,n-c+1) is the 
			http://blog.csdn.net/linulysses/article/details/5634104
}

��һ�ִ��� 'b' ɾ�� �� 'a' ��� 'aa' in O(1) space, ����ԭ�����㹻�ռ�任����´�{
	==> solution 2: ��β��ʼ replace. 
	==> solution 1: ��ɨһ����� 'b', 'a'	����������´����ȣ��ٴ�β��ʼ
}

Search in Rotated Sorted Array{
	Suppose a sorted array is rotated at some pivot unknown to you before hand.
	
	(i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
	
	You are given a target value to search. If found in the array return its index, otherwise return -1.
	
	==> If no duplicate exists in the array: O(lgn), ���� A[s] A[c] A[e] �Ĺ�ϵ���ж�������һ��
	==> If duplicate exists in the array: �����������ʱ�������ж�����, ��������Կɷֶ�
}

Given a array of integers , find 3 indexes i,j,k such that, i< j < k and a[i] < a[j] < a[k] {// ???
	==> Solution: O(lg n), two candidate index set, with s2[0] < s1[0] < s1[1], for any i, if(a[i] > s1[1]){found!}else{if(a[i] > s2[0]){replace s1 with s2[0] and a[i], clear s2;}
	}
}

һ������������һ��local min(�������Ҷ�С)�Ϳ���{// ???
	����1,2,3,7,19,8,6,11,34,23,67 �����6��8,11С��23��34,67С������ local min��
	==> solution O(lgN) ���ظ�����: �� b > m < e, if(m-1 < m) search [b,m] else search [m-1,e]
}

��һ��int������������������������ڵ��������������С�ڵ����Ҳ�������{// ���˵�����ʲô?
	==> solution: ������ɨ�裬�� max index �� stack; ���ҵ���ɨ�裬���� min ���� stack ��� max �Ƚ��Ƿ�ͬһ index
}

��������N����N^7С������Ҫ����㷨��O(n){// ������ʾ..˵��N�����Ƿ����룬bucket sort, ע�� 7 �ǳ���
	==> solution: ��ͷɨ�裬���� N^6 ������Ƶ�����; �ٴξ��� N^5, N^4, ... N, һ�־��� O(n); ���·ֶε�ַ������һ��, ��7��
}

����������ɵļ������ҵ����ȴ���=3����ĵȲ�����{// ???
	eg. ����[1,3,0,5,-1,6], ���[-1,1,3,5]. Ҫ��ʱ�临�Ӷȣ��ռ临�ӶȾ���С
}

count the number of reverse order pair{
	Give an integer array A[] with size of n, count the number of reverse order pair.
	e.g. [6 2 5 3 9], pairs: (6,5), (6,3), (5,3) count = 3;
	==> Solution: insertion sort/bubble sort + count O(n^2)
	==> Extension: give B[] and A[], count the number of reverse order of A[]'s elements in B[]'s order
}

find the kth integer that contains only factors 2,3,5, starting from 1 {
	==> solution 1 : O(k), 3 queue, each for 2,3,5. find the minimum of the three, if it pop up from the small number, then multiple with number that greater than it // ע�����
		
	int FindKthElement(vector<int>& Factors,int k)// for more general number of factors
	{
		vector< queue<int> > Qs;
		vector<int> IndQ;
		sort(Factors.begin(),Factors.end());
		for(int i = 0; i < Factors.size(); ++i)
		{
			queue<int> tempQ;		
			tempQ.push(Factors[i]);
			Qs.push_back(tempQ);
			tempQ.pop();
			IndQ.push_back(i);
		}
		
		int result = 1;
		while(--k > 0)
		{
			result = Qs[IndQ[0]].front();
			Qs[IndQ[0]].pop();
			for(int i = IndQ[0]; i < Factors.size(); ++i)
			{
				Qs[i].push(result*Factors[i]);
			}
			int CurInd = 0;// maintain the min-heap
			while(CurInd < Factors.size())
			{
				int Left  = (CurInd + 1) * 2 - 1;
				int Right = (CurInd + 1) * 2 ;
				int Min   = CurInd;
				if(Left < Factors.size())
					if(Qs[IndQ[Min]].front() > Qs[IndQ[Left]].front()) Min = Left;
				if(Right < Factors.size())
					if(Qs[IndQ[Min]].front() > Qs[IndQ[Right]].front()) Min = Right;
				if(Min != CurInd)
				{
					int temp = IndQ[CurInd];
					IndQ[CurInd] = Min;
					IndQ[Min]    = IndQ[CurInd];;
					CurInd = Min;
				}else
					break;
			}
		}
	
		return result;
	}
	==> solution 2: [1,k*2] ����һ���� i��ʹ�� f(i) := i/2 + i/3 + i/5 - i/6 - i/15 - i/10 + i/30 < k but f(i+1) >= k?
}

Maximum profit from stocks {
	http://www.leetcode.com/groups/google-interview/forum/topic/maximum-profit-from-stocks/
	Given M different stocks prices across N different day and an Amount D dollars. how do you maximise your profit by buying and selling stocks. more formally: this is an MxN matric. Each row corresponds to a stock and N columns correspond to price of the stock on day ��i��. You are given D dollars to buy and sell stocks. maximize your profits.
	==> If can buy and sell daily: for each stock, calculate p_{i+1}/p_i to get a M*N matrix; then for each day, find the max(d)
	==> If one can only buy and sell once: for each stock, find the p_max/p_min; find the stock with the max p_max/p_min
		Best Time to Buy and Sell Stock{// ֻ��һ�ν��ף���֮ǰ��͵�
		    int maxProfit(vector<int> &prices) {// 2013.09.23 �޴�ͨ��
		        if(prices.empty()) return 0;
		        int Profit = 0;
		        int PreMin = prices[0];
		        for(int i = 1; i < prices.size(); ++i){
		            Profit = max(Profit,prices[i]-PreMin);
		            PreMin = min(PreMin,prices[i]);
		        }
		        return Profit;
		    }
		}    
		Best Time to Buy and Sell Stock II{// ���Զ�ν��ף������ͬʱ��һ����Ʊ
		    int maxProfit(vector<int> &prices) {// 2013.09.23 �޴�ͨ��
		        int n = prices.size();
		        if(!n) return 0;
		        int Profit = 0;
		        int PreMin = prices[0];
		        for(int i = 1; i < n; ++i){// for every 
		            if(prices[i] < prices[i-1]){
		            	Profit += max(prices[i-1]-PreMin,0);
		            	PreMin = prices[i];
		            }
		        }        
		        return Profit+max(prices[n-1] - PreMin,0);
		    }
		}  
		
		Best Time to Buy and Sell Stock III{// ������ཻ�����Σ�����ͬʱ����
			==> ������ for each i, �����һ���� i �����Ľ��� maximum profit; ���ҵ��� for each i, ����ڶ��ν����� (i,e] �ڷ����� profit
			// Error 1: �������ʱ���� PreMin, ӦΪ PreMax, ���� PreMax Ӧ��ǰ
		    int maxProfit(vector<int> &prices) {// 2013.09.23 �ö�̬�滮���������Ӵ���ǰ��ɨ��
		        int n = prices.size();
		        if(n <= 1) return 0;
		        int* Profit = new int[n]; Profit[0] = 0;
		        int PreMin = prices[0]; 
		        for(int i = 1; i < n; ++i){
		            Profit[i] = max(Profit[i-1],prices[i]-PreMin);
		            PreMin = min(PreMin,prices[i]);
		        }
		        int MaxProfit = 0; //max(Profit[n-1],Profit[n-2]);
		        int PreMax = prices[n-1];
		        for(int i = n-2; i > 0; --i){
		        		PreMax = max(PreMax,prices[i]);
		        		MaxProfit = max(MaxProfit,PreMax - prices[i] + Profit[i-1]);        		
		        }
		        MaxProfit = max(MaxProfit,Profit[n-1]);
		        delete[] Profit;
		        return MaxProfit;
		    }
		} 
}

Two Sum Problem{
	==> solution 1: sort O(nlgn) and find O(n) with space complexity O(1), ��ͷɨ��
	==> solution 2: hash(a_i) and hash(sum - a_i) with time complexity O(n) but space complexity O(n);
	==> Extension: �� abs-sorted �� array ���� two sum: two pointer��һ��������һ������
	==> Extension: ������ Ϊ k 
}

Three Sum, without extra space{
		* Solution 1: O(n^{k-1}), similar to K-sum: �� hash table, Ȼ��� -(a[i] + a[j])�� value; 
		* Soultion 2: Using extra space, C_n^{k/2} and O(n^{k/2} lgn) if k is even; C_n^{(k+1)/2} and O(n^{(k+1)/2}) if k is odd. Using extra space ����
		* Solution 3: O(n^2): �� sort O(nlgn), Ȼ��ÿ���������ܷ��ҵ�������ʹ����֮Ϊ 0 O(n^2)
		* Conclusion: for K-sum, requires O(n^{Omega(K)}) <On the Possibility of Faster SAT Algorithms> by Mihai Patrascu and Ryan Williams.
}

N sum problem{
		* solution 1: recursive enumerate
		* solution 2: backpack problem 
		* extension : ����������a,b����С��Ϊn,����Ԫ�ص�ֵ��������������Ҫ��ͨ������a,b�е�Ԫ�أ�ʹ����aԪ�صĺ�������bԪ�صĺ�֮��Ĳ���С��
}

����string, �������ǵ������ȳ���substring, �������ǵľ���Ϊdistance=sum_i(s1[i]-s2[i])���Ҿ�����������substring{// DP 
	==> If without absolute "s1[i]-s2[i]"
	==> If with absolute "abs(s1[i]-s2[i])"
	==> Solution DP or ���: O(n*m) time + O(n) space
	f(str1,str2) = max{f(str1-1,str2), f(str1,str2-1), g(str1-1,str2-1)+ str1[end] - str2[end]},where g(str1,str2) is the max distance of suffix. 
}

һ�����ʵ��б�Ҫ�ҳ��������ʣ�����û����ͬ����ĸ���֣����ҳ��ȳ˻����{
	==> Solution: ���ݵ����б���һ�� 26 �����. ��һ�� 'A' yes or no, Node{int MaxLen} ����ÿ�� word �� signature, bool IsInWord[26]; ֻ����ϸ� signature ������ĳ��ȡ��м�ڵ�� MaxLen ��ʾ���ӽڵ��е�����ȡ� Ȼ���� complementary �ܹ� O(n)
}

�� string s ����� lexicographic order �������ظ���ĸ���Ӵ�{ eg: Input "babab"; Output "ba"
==> solution: ���� s signature bool IsInWord[26], �����͵ó����� substring len = sum(IsInWord). 
}
Longest Substring Without Repeating Characters {
	�� A[] ������Ӵ���ʹ֮�������ظ���ĸ���� Begin[26], �� startInd, ֻ���� Begin[26] > startInd �Ÿ��� MaxLength.
	==> ˼·: �� C[26] ��¼��ٳ��ֵ� index, Pre ���ظ��ִ����ַ�ǰ�����꣬�������� A[Pre]
    int lengthOfLongestSubstring(string s) { 
        int C[26], Max = 0, Time = -1, i;
        for(i = 0; i < 26; ++i) C[i] = -1;
        for(i = 0; i < s.length(); ++i){
            int j = s[i] - 'a';
            if(C[j] > Time){
            	Max = max(Max,i-Time); 
            	Time = i;
            }
            C[j] = i;
        }
        return Max;
    }
}


��minimum window in A, contains string B {// ע�� ���д����
input
A = [1, 9, 3, 4, 12, 13, 9, 12, 21]
B = [9, 12, 21]

output
A[6..8] = [9, 12, 21]

����һ�� B.length() �������飬��¼��һ��Ԫ���� B[i] �� window �� A �� �ʼԪ��λ��. �ٽ�һ�� vector<int> Set[26], Set[i] ��Ӧ 'a'+i �� B �г��ֵ�Ԫ��λ��
==> ÿ���ⶼ�� BeginInd, �� NextElementInB ����
==> One-pass solution: maintain a queue Q with element <BeginInd,NextElementInB>, initially, it has an element <-1,0> 
	scan A one-pass, for each element A[CurInd], update Q with
	  1. if A[CurInd] == B[NextElementInB]; if (NextElementInB == 0){ BeginInd = CurInd}; update NextElementInB += 1; 
	  2. Delete q[b1,n1] if existing q[b2,n2] in Q, s.t. b1 <= b2 and n1 <= n2;
}

Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n){
	// ����������� ��ĸ����С�����������ܴ�, ����������޹�
	==> solution: 1. �� cnt[26/52] ͳ�Ƴ� T ���ֵ��ַ��������ַ�������� posCnt, negCnt = 0; 2. right=0 <n ����cnt posCnt negCnt ʹ posCnt = 0; left = 0 < right ����cnt posCnt negCnt ʹ negCnt = 0; 3. �� posCnt = negCnt = 0 �����ҵ����� T �� S �Ӵ�, ���� left ����һ�� T ���ַ������ҵ��˶�����Ӵ�, ���� Max[2](��ʼ�볤��)����ֻ����� T, ��ֻ��Ҫά�� posCnt == 0
	For example,
	S = "ADOBECODEBANC"
	T = "ABC"
}

compute the lexicographically smallest permutation of [1,2,....n] given a signature{//
	signature: eg {3,2,1,6,7,4,5} ==> signature = "DDIIDI" --> ��С permutaion = 3,2,1,4,6,5,7;
	==> solution O(n): 1. from "DDIIDI" get I[7] = [2,1,0,0,1,0,0] ��β��ͷ�� 'D' �� 1���� 'I' �� 0. 
		2. �� Min = 1; ��ʼÿ������ 'D' �� nextMin = I[i] + Min + 1; I[i] = Min + I[i]; ֱ���������һ�� I[i] == 0 ��reset Min = nextMin; ���� 'I' �� I[i] = Min; Min += 1;
}


Given an int array A[n], higher value more candy than neighbor, every A[i] at least got a candy, find the minimum number of candy{
	==> c[i] Ϊ i ���ǹ������� c[i] �� min{ > i �����ݼ��� i �������Ŀ ��< i ��Ԫ�������������� i �������Ŀ}
	���Է�����ɨ��c[] ��ʼ��Ϊ 1
1. ��һ���ǰ���ɨ�ϸ񵥵������䣬c[i] = c[i-1]+1;
2. �ڶ���Ӻ���ǰɨ�ϸ񵥵������䣬c[i] = max(c[i],c[i+1]+1);����c[i]�ǵ�һ��ɨ��������ĸ�ֵ��c[i+1]+1�Ǳ���ɨ��������ĸ�ֵ���ƺ�ֻ�ÿ����Ϲյ�Ϳ����ˣ�
}


Rotation an array A[n] k position with O(1) space and O(n) time{
	==> �򵥵Ľⷨ reverse the whole A[1-n], then reverse A[1-i], A[i+1, -- n] separately
}

Largest Rectangle in Histogram{
	==> O(n) with stack; stack ��ǰ�ߵ�Ԫ�� height �� index �������͵���ͳ��֮ǰ��. Ϊ��Ӧ�� input ������Ԫ�� 0
}

Trapping Rain Water {
	==> �� Largest Rectangle in Histogram ���Ӧ���׷�����: 1. ����߿�ʼ bar; 2. ���� bar ����bar ����ΰ�? ά�� stack.front() ΪĿǰ��� bar
}

Container With Most Water{// �븴����, Ӧ���� 
	Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.
==> Solution: i= 0, j = n - 1; ��ô��ǰ��area��min(height[i],height[j]) * (j-i); while (i < j){ if(height[i] < height[j])++i; else --j}
}

Maximal Rectangle���� 0,1 ������ 1 ���ɵ�����Ӿ���{
	==> solution 1 O(n^3): p(i,j) �õ㿪ʼ������ÿ�б���ÿ�㿪ʼ��������, �� k >= i; m = min(l_k,l_{k+1}}, �� m  Ϊ0 �� k = n-1 ʱͣ
	==> solution 2 O(n^2): ���� Largest Rectangle in Histogram�� ��ÿ�����Ը���Ϊ�׵�����Ӿ��� r = 0-->n-1��ÿ�� O(n). ���Ӹ߶ȼ�λ�ø�����ѭ����. ע���ڸ�������ʱ�������ӵ� index Ҫ�ĳ�pop �������ӵ�
}


Maximun sub matrix n*n (2d version of maximum subarray){// ����һ���
	==> ˼��: 1. �����ظ�����; 2. ����ά���ɵ�ά
	==> Blue force O(n6)
	==> solution O(n3): 1. for each column, calculate prefix sum O(n^2); 2. for each pair of rows (rs,re) and each column c calculate f(rs,re,c) as the sum from rs to re in column c. O(n^3); 3. For each (rs,re,*), find the maximum subarray. O(n^3)
	==> extension: d-dimension matrix, maximum submatrix O(n^{2*d-1})
}

�㹲��m������ռ䡣��n�����󣬵�i���������ʱ��Ҫ R[i]���ռ䣬��������ҪO[i]���ռ�(����O[i]<R[i]), ������n�������˳��ʹ���������������{
	e.g. ����㷨ҲӦ���ܹ��жϳ�������ζ����ܴ������������ȷ�˵��m=14��n=2��R[1]=10��O[1]=5��R[2]=8��O[2]=6
==> solution ����������(��������) i,j. If run i first: O_i + R_j; if run j first: O_j + R_i; then if O_i + R_j < O_j + R_i, run i first. i.e. run max(R-O) first. 
}

Having an [infinite] stream of numbers write a function to take an element with equal probability for each{
	��Ȼ�����ޣ��������ǵñ�֤ǰ N ��Ԫ��ÿ�������� 1/N, ��ʵֻҪ��֤�� N ��Ԫ��(����)���� 1/N ���ʱ�ѡ�����������ǣ�  
	����һ����ǰԪ�أ������鳤�� N��ÿ�� 1-1/N �Ļ��� ���� ���� ǰ������ stream ֹͣʱ������������ѡ�� number  
}

��2*N���ļ����ļ��Ĵ�С������size[2*N]�С�Ȼ����Ҫ�ֳ�N��(ÿһ�ݿ�����1���߶���ļ�)��Ҫʹ��N���е��ļ�size֮�͵����ֵ��С{
	==> ���ƽ�: sortһ�� �Ӵ���С��n�� Ȼ��ʣ��n�����Ӵ���С�����������߶���С���ļ���ȥ
}

��������õ�����  �����С��m��pair{//
	1) insert A[0] B[0]
	2) pop up the smallest, insert A[1]B[0],A[0]B[1]
	3) pop up the smallestA[i]B[j] , insert A[i+1]B[j],A[i]B[j+1]... 	until u get m
	==>  �е��� Young matrix�����Խ��߽��� O(m)
}

Given an array A of N integers, where A[i] is the radio at i, find the number of intersecting pairs{
	e.g. A = {1,5,2,1,4,0} ���� 11 ��
	==> solution: time O(n) + space O(min(n,radio)). ��һ��ɨ�� c[i] ��ʾ ��ߵ��� i �ĸ��� (������ i ����). �ڶ��� sum up: m(i) := min(i,A[i]); sum += m(i) + c[A[i]-m(i)] 
}

Given an unsorted array of integers, find the length of the longest consecutive elements sequence.{
	==> O(n): 1. ����Ԫ���� unordered_set, 2.��ȡһԪ�� v, �� v+1 up, v-1 down �� set �в��ң���������Ԫ�� count 1 ����ȥ 
}

������������и���������������ǰ�棬���������͸���ԭ�������˳�򲻱� inplace ʱ�临�Ӷ�Խ��Խ��{// ???
	eg: input -5 2 -3 4 -8 -9 1 3 -10;  output -5 -3 -8 -9 -10 2 4 1 3.
	==> ���������ڰ�һƪ�����Ե���Ϊ��λͷβ����. nlogn
}

Palindrome Partitioning{// given a string, �ֽ�� ���� palindrome strings. (1) ������п��ܵ���� (2) �ҳ� substring ��Ŀ���ٵ��и� 2013.09.13
	--> ˼· DP: �����ά bool p(i,j), substring[i,j) is palindrome. �ݹ� p(i,j) = p(i+1,j-1) && a[i] == a[j-1]; �Ӿ������½ǿ�ʼ; �߽� p(i,j) = true if i+1 >= j; 
		vector<int> Split(bool * p, int size){
		}
		
    vector<vector<string> > partition(string s) {
    	vector<vector<string> > result;
    	if(s.empty())return result;
    	int n = s.size();
      if(n == 1){
      	result.push_back(s);
      	return result;
      }
      vector< vector <bool> > p;
      for(int k = 0; k < n-1; ++k){
      	vector<bool> t;
      	t(n-k-1,false);
      	t[0] = s[k] == s[k+1];
      	if(k < n-2) t[1] = s[k] == s[k+2];
      	p.push_back(t);
      }
      
      for(int k = 0; k < n-1; ++k){	
      	for(int i = 0; i + k < n; ++i){
      		p[i*n+i+k] = s[i] == s[i+k];
      		if(k > 1) p[i*n+i+k] &&= p[(i+1)*n+i + k-1];
      	}
      }
      for(int i = 1; i < n; ++i){
      	
      result = partition(string(s.begin()+1,s.end()));  
    }	
}

Reverse Nodes in k-Group{// 2013.09.23 VC ͨ�������޷�ͨ�� LeetCode
    int reverseKGroupSub(ListNode*&head,int k,ListNode*&PreEnd){// return reverse number
        if(k <= 1 || head == 0) return 0;
        int SwapNum = 1;
        ListNode * CurrNode = head;
        ListNode * NextNode = CurrNode->next;
        while(--k > 0 && NextNode){
            ListNode * temp = NextNode->next;
            NextNode->next  = CurrNode;
            CurrNode = NextNode;
            NextNode = temp;
            ++SwapNum;
        }
        PreEnd = head;
        head->next = NextNode;
        head = CurrNode;
        return SwapNum;
    }
    ListNode *reverseKGroup(ListNode *head, int k) {// 2013.09.23
        ListNode PreHead(0); 
        PreHead.next = head;
        head = &PreHead;
        while(head->next){
            ListNode* PreEnd;
            int ReverseNum = reverseKGroupSub(head->next,k,PreEnd);
            if( ReverseNum <= 1) break;
            if( ReverseNum < k){
                reverseKGroupSub(head->next,ReverseNum,PreEnd);
                break;
            } 
            head = PreEnd;
        }
		return PreHead.next;
    }
}

Remove Element that equal to a given value{ // LeetCode ����ͨ�� [2,2,3], 2, �� VC ����û������
  int removeElement(int A[], int n, int elem) {// 2013.09.23 
      if(n == 0) return 0;
      int s = A[0] == elem? 0:1;
      for(int i = 1; i < n; ++i){
      	if(A[i] != elem)A[s++]  = A[i];
      }
      return s;
  }
}

Remove Duplicates from Sorted Array{
    int removeDuplicates(int A[], int n) {// 2013.09.23 �޴�ͨ��
        if(n == 0) return 0;
        int newLen = 0;
        int CurInd = 1;
        while(CurInd < n){
            if(A[CurInd] != A[newLen])A[++newLen] = A[CurInd];
            ++CurInd; 
        }
        return newLen+1;
    }
}

Remove Duplicates from Sorted List{// 2013.05.02 8:06pm 8:09pm �޴�ͨ��
    ListNode *deleteDuplicates(ListNode *head) {
        if(!head) return NULL;
        ListNode* pre = head;
        while(pre->next){
            if(pre->next->val == pre->val){
                pre->next = pre->next->next;
            }else{
                pre = pre->next;
            }
        }
        return head;
    }	
}

Remove Duplicates from Sorted List II{// 2013.05.02 8:10pm VC ���� 8:37pm 
    ListNode *deleteDuplicates(ListNode *head) {
        if(!head) return NULL;
        ListNode tempN(0);
        tempN.next = head;
        ListNode*pre = &tempN;
        while(pre->next){
            ListNode*cur = pre->next;
            while(cur->next){
                if(cur->next->val == pre->next->val){
                    cur = cur->next;
                }else{
                    break;
                }
            }
            if(cur == pre->next){
                pre = pre->next;
            }else{
                pre->next = cur->next;
                // ����: pre = cur; ���һ��
            }
        }
        return tempN.next;        
    }	
}
    
Reverse Linked List II{// 2013.05.02
    ListNode *reverseBetween(ListNode *head, int m, int n) {// ��Ҫ VC �Ҵ���
        if(m >= n || !head) return head;
        ListNode tempNode(0);
        ListNode* Pre1st = &tempNode;
        n = n-m; // number of reverse
        
        Pre1st->next = head; // �ؼ�����һ��ͷ������ head reverse ������
            
        while(--m && Pre1st->next != NULL){// guarantee Pre1st->next is not empty
            Pre1st = Pre1st->next;
        }
                
        ListNode*start = Pre1st->next;
        ListNode*next  = start->next;
        while(next && n-- > 0){
            ListNode* temp = next->next;
            next->next = start;
            start  =  next;
            next = temp;
        }
        Pre1st->next->next = next;
        Pre1st->next = start;
        return tempNode.next;// ���� Pre1st->next;���� Pre1st �����Ѿ��ı�
    }
    
    // 2013.08.04 ������ɽ��� n-m ��, �������ˣ���
    ListNode *reverseBetween(ListNode *head, int m, int n) {
	    if(n-m < 1) return head;
	    
	    ListNode* H, *T;        
	    int i = n- m;        
	    H = head;
	    while(--m > 0)H = H->next; T = H;
	    reverseBetween(H->next, 0, i-2);
	    while(i-- > 0)T = T->next; 
	    int v = H->val; H->val = T->val; T->val = v;
	    return head;
	}
}

Reverse Nodes in k-Group{// �����Ĳ��ý��� // �޴�ͨ��! 2013.05.02
    ListNode *reverseKGroup(ListNode *head, int k) {
        if(!head || k <= 1) return head;
        ListNode tempN(0);
        ListNode* Pre1st = &tempN;
        Pre1st->next = head;
        while(Pre1st->next){
            int i = 1;
            ListNode* temp = Pre1st->next;
            while(temp->next&&i<k){// detect whether should go to next loop
                temp=temp->next;
                ++i;
            }
            if(i<k) break;
            ListNode* start = Pre1st->next;
            ListNode* next = start->next;
            
            while(--i && next){
                temp = next->next;
                next->next = start;
                start = next;
                next = temp;
            }
            Pre1st->next->next = next;
            ListNode* NextPre1st = Pre1st->next;
            Pre1st->next = start;
            Pre1st = NextPre1st;
        }        
        return tempN.next;        
    }
}

Remove Nth Node From End of List{// �޴�ͨ�� 2013.05.02, �� 2013.09.21 �д�
    ListNode *removeNthFromEnd(ListNode *head, int n) {
        if(!head || n <= 0) return head;
        ListNode PreHead(0);
        PreHead.next = head;
        ListNode* end = &PreHead;
        while(n-- && end->next){
            end = end->next;
        }
        ListNode*pre = & PreHead;
        while(end->next){
            end = end->next;
            pre = pre->next;
        }
        
        pre->next = pre->next->next; // delete pre->next from the linked list
        return PreHead.next;
        
    }
}

Partition List{// VC ��� 2013.05.02
// ����һ: ���� linked list �����㣬����ʱ�� pre->next->next �����鷳
// �����: �㷨��ƴ���, ���ǽ������㣬���Ǹı��ĸ�����

    ListNode *partition(ListNode *head, int x) {
        if(!head)return NULL;
        ListNode L(0),GE(0);
        ListNode* pL = &L, *pGE = &GE;
       	while(head){
       		ListNode* t = head;
       		head = head->next;
       		if(t->val < x){
       			pL->next = t;
       			pL = t;
       		}else{
       			pEG->next = t;
       			pEG = t;
       		}
       		t->next = 0;
       	}
       	if(L.next != 0){
       		pL->next = Ge.next;
       		return L.next;
       	}else{
       		return GE.next;
       	}
    }
}

Merge Two Sorted Lists{// ���� && д�� & 2013.05.02 // �ж�ָ���Ƿ�Ϊ null ʱȡ����, e.g while(!p){}; ӦΪ while(p){}
    ListNode *mergeTwoLists(ListNode *l1, ListNode *l2) {
        ListNode tempN(0);
        ListNode* pre = &tempN;
        while(l1&&l2){
            if(l1->val > l2->val){
                pre->next = l2;
                l2 = l2->next;
            }else{
                pre->next = l1;
                l1 = l1->next;
            }
            pre = pre->next;
        }
        pre->next = l1?l1:l2;
        return tempN.next;
    }
}

Merge k Sorted Lists{// �� VC �������� 2013.05.02
    ListNode *mergeKLists(vector<ListNode *> &lists) {
        ListNode PreNode(0);        
        deque<ListNode*> MinHeap;        
        for(int i = 0; i < lists.size(); ++i){
            if(lists[i]){
                MinHeap.push_back(lists[i]);                
            }
        }
        int size = MinHeap.size();
        for(int i = size/2; i >= 0; --i){// heap initialize
            RearrangeMinHeap(MinHeap, i);
        }
        ListNode*pre = &PreNode;
        while(size>1){
            pre->next = MinHeap[0];
            MinHeap[0] = MinHeap[0]->next;
            if(!MinHeap[0]){
                MinHeap[0] = MinHeap[size-1];// ����: MinHeap.pop_front(); �ı� size ��û�����ţ��й� MinHeap ����
                MinHeap.pop_back();
                --size;
            }
            RearrangeMinHeap(MinHeap, 0);
            pre = pre->next;
        }
        pre->next = (size == 0? NULL:MinHeap[0]);// ���� Ϊ size==1
        return PreNode.next;
    }
    
    void RearrangeMinHeap(deque<ListNode*>& MinHeap, int i){
        int size = MinHeap.size();        
        do{
            int l = 2*(i+1)-1;
            int r = 2*(i+1);
            int m = i;
            if(l < size){
                m = MinHeap[l]->val < MinHeap[m]->val?l:m;
            }
            if(r < size){
                m = MinHeap[r]->val < MinHeap[m]->val?r:m;
            }
            if(m != i){
                ListNode*temp = MinHeap[m];
                MinHeap[m] = MinHeap[i];
                MinHeap[i] = temp;
                i = m;
            }else{
                break;
            }
        }while(1);
    }	
}

Rotate List{//û����� 2013.05.02 10:42pm
	    ListNode *rotateRight(ListNode *head, int k) {
        if(!head || k <= 0) return head;                
        ListNode*last = head;
        while(k-- && last->next){
            last = last->next;
        }       
        if(last==head) return head;
        
        ListNode tempN(0); tempN.next = head;
        ListNode*prev = &tempN;
        while(last->next){
            prev = prev->next;
            last  = last->next;
        }
        last->next = head;
        tempN.next = prev->next;
        prev->next->next = NULL;
        return tempN.next;
    }
}

Regular Expression Matching with support for '.' and '*'{// 
	==> ע��� "*" ������, ��ȷ�Ǳ�ʾ��"ǰ��"�ַ��������(���� 0)���ظ�
	==> Do NOT working: �� DP, �� str �� pattern ��������ִ�, ��Ҫ������ recursive, ��Ϊ complexity Ϊ exponential. pattern ���� ���Ψһ, eg. b*b ���Ϸ����� bb* �Ϸ�
	
	bool isMatch(const char *s, int slen, const char *p, plen){
		int si = 0, pi = 0;
		while(si < slen && pi < plen){
			if(s[si] == p[pi] || p[pi] == '.'){
				++si,++pi;
				if(pi < plen && p[pi] == '*'){
					while(s[si-1] == s[si] && si < slen) ++si;
					pi += 1;
				}
			}else if(pi + 1 < plen && p[pi+1] == '*'){
				pi = pi + 2;
			}else{
				return false;
			}
		}
		if(pi == plen) return si == slen;
		return false;
	}
}

ƥ������ ͨ��� * and ?; e.g. ��J* Smi??�� ����ƥ�䡰John Smith��{
==> solution: �ֿ� *, ? , ������� ���� how to handle J*ss Jesass, 

bool IsFit(const char* txt, int txtSize, const char* pat, int patSize){// recursive, ���� const char* ���ܸ�ֵ����char*
	int patPos,shift;
	if(txtSize == 0){// empty text
		if(patSize == 0) return true;
		for(patPos = 0; patPos < patSize; patPos++) if(pat[patPos] != '*') return false;
		return true;
	}
	if(patSize == 0) return false; // non-empty text but empty pattern	
	if(txt[0] == pat[0])return IsFit(txt+1, txtSize-1, pat+1, patSize-1);
	
	patPos = shift = 0;	
	bool HasStar  = (pat[0] == '*'); 
	while((pat[patPos] == '*' || pat[patPos] == '?') && patPos < patSize){// until next pat[patPos] != '?', '*'
		shift += (pat[patPos] == '?'?1:0);
		HasStar |= (pat[patPos] == '*');
		patPos++;
	}
	
	if(txtSize < shift || patPos == 0) return false; // not enough txt or pat[patPos] != '*','?'; ���� forget "|| patPos == 0"
	bool result = IsFit(txt+shift, txtSize-shift, pat+patPos, patSize-patPos);
	if(!HasStar)return result;// if no '*', do not need to consider shifting
	if(patSize == patPos) return true; // since HasStar == true;
	
	do{
		while(txt[shift] != pat[patPos] && shift < txtSize) shift++;// find the next starting index
		if(shift == txtSize) return false;// no result
		if(IsFit(txt+shift+1,txtSize-shift-1,pat+patPos+1,patSize-patPos-1)) return true; // return the shift result
	}while(1);
}

}

String Matching Problem{
==> Solution 1: Knuth�CMorris�CPratt algorithm(KMP): build index for search pattern, p[i] is that if i th charactor fails, which index should go to 
==> Solution 2: Trie tree: prefix tree for suffix, good for large number of small patterns, 
==> Solution 3: Rabin�CKarp hash: build hash functions (rolling hash, fixed lengths as the searched patterns) for patters and each substring of the text. good for search multiple patterns. e.g f(str) = str[0]+str[1]*p+...+ str[k-1]*p^(k-1), where p is a large prime
}

copy LinkedList{ where m_pSibling ָ�������е���һ������NULL�������C++�������£�
	 struct ComplexNode
	{
	    int m_nValue;
	    ComplexNode* m_pNext;
	    ComplexNode* m_pSibling;
	};

	==> Solution: copy each oldnode (recursively, then next can be copied to new node), with the newnode->next = oldnode, and oldnode->next = newnode; 2. update noldnode->next->sibling = oldnode->next->next ���µ�� sibling update, 3. �µ�Ӿɵ��н��ѳ��� 
}

reverse Polish Expression{
	1 + 2 * 3 = 2 3 * 1 +;
	==> �ø��� stack ������ÿ����һ��������� stack ��ȡ����������
}

���������������Ŀ{
1.��������������Ƿ��л�
	==> Solution: ʹ������ָ��p1,p2������ͷ��ʼ������p1ÿ��ǰ��һ����p2ÿ��ǰ�����������p2��������β����˵���޻�������p1��p2��Ȼ����ĳ��ʱ������(p1==p2)���Ӷ���⵽�������л���

2.��������������(head1, head2)��������������Ƿ��н��㣬����з��ص�һ������, ע�� L1, L2 ������ circle. 
	==> Solution: ���head1==head2����ô��Ȼ�ཻ��ֱ�ӷ���head1�����򣬷ֱ��head1,head2��ʼ���������������䳤��len1��len2������len1>=len2����ôָ��p1��head1��ʼ����ƶ�len1-len2����ָ��p2=head2������p1��p2ÿ�����ǰ��һ�����Ƚ�p1p2�Ƿ���ȣ������ȼ����ظý�㣬����˵����������û�н��㡣


3.����������(head)������л��Ļ��뷵�ش�ͷ�����뻷�ĵ�һ���ڵ㡣
	==> ���ǿ��Լ���������Ƿ��л�������л�����ôp1p2�غϵ�p��Ȼ�ڻ��С���p��Ͽ���������Ϊ��p1=p, p2=p->next, p->next=NULL����ʱ��ԭ��������Կ�������������һ����head��ʼ����һ����p2��ʼ��������������ķ����������ҵ����ǵĵ�һ�����㼴Ϊ����

3.2 �ж�һ�� linkedlist �Ƿ��� palindrome

3.3 ��һ�� linkedlist ���۵�����任��һ���� linkedlist

// ����������Խ��ɽڵ�ת�����½ڵ�
4.ֻ������������ĳ�����p(�������һ����㣬��p->next!=NULL)ָ�룬ɾ���ý�㡣
	==> �����Ƿ�p������,Ȼ��p->next������copy��p�У�������ɾ��p->next���ɡ�

5.ֻ������������ĳ�����p(�ǿս��)����pǰ�����һ����㡣
  ==> �취��ǰ�����ƣ����ȷ���һ�����q����q������p�󣬽�������p�е�����copy��q�У�Ȼ���ٽ�Ҫ��������ݼ�¼��p�С�	
}

// �����õ��ַ����� signature
Given a string S and a list of words (same length) L. Find all starting indices of substring(s) in S that is a concatenation of each word in L exactly once and without any intervening characters{
	==> solution: O(n) 1. �� unordered_set ʹ��ѯ�Ƿ��� list ��Ϊ O(1); 2. DP ά�� word.size() �� deque<bool> queue��Q[i] ��ʾ�� i ��ʼ�Ѿ����ֵ� words, ���� vector<bool> Appear(word.size(),false)
}

Substring with Concatenation of All Words{ 
     vector<int> findSubstring(string S, vector<string> &L) {// L is not empty
        // �������� S:"a"; L: ["a","a"], ��Ҫ�õ� unordered_multiset
        vector<int> result;
        if(L.size() == 0 || L[0].size() == 0 || L[0].size() > S.length() ) return result;
        int l = L[0].size();
        vector<int> Int(l,-1);
        unordered_map<string,vector<int> > m;
        for(int i =  L.size()-1; i >= 0; --i)m[L[i]] = Int;
        for(int curInd = 0,i = S.length()-l; i >= 0; --i){
            unordered_map<string,vector<int> >::iterator it = m.find(S.substr(i,l)),j;
            if(it == m.end()){
                for(j = m.begin(); j != m.end(); ++j) j->second[curInd] = -1;
            }else if(it->second[curInd] == -1){
                it->second[curInd] = i;
                int cnt = 0;
                for(j = m.begin(); j != m.end(); ++j) cnt += j->second[curInd] != -1;
                if(cnt == m.size()) result.push_back(i);
            }else{
                for(j = m.begin(); j != m.end(); ++j) j->second[curInd] = j->second[curInd] > it->second[curInd]? -1:j->second[curInd];
            }
            if(++curInd == l) curInd = 0;
        }
        return result;
    }
}

You have k lists of sorted integers. Find the smallest range that includes at least one number from each of the k lists.{
	==> solution: update the array with the smallest first element
	For example,
	List 1: [4, 10, 15, 24, 26]
	List 2: [0, 9, 12, 20]
	List 3: [5, 18, 22, 30] 
}

very large byte stream (PB) synchronization algorithm{
	given:
	unsigned char read_byte(); �� side effect that it advances a byte pointer in the stream
	write:
	unsigned char read_sync_byte(); �� may result in >1 calls to read_byte()
	
	remove byte '03' from the stream if the stream is in pattern 00 00 03
	read_byte():
	00 0f 42 17 00 00 03 74 00 00 00 00 14 ...
	read_sync_byte():
	00 0f 42 17 00 00 74 00 00 00 00 14
	
	==> solution: �� ring buffer, Buffer[0-5], start, len; if str[cur] == pattern[len], ++len; else read out; if len == pattern,
	==> solution: int pre00cnt = 0; if( byte == '00'){pre00cnt +=1; if(pre00cnt == 3) pre00cnt = 2;} else pre00cnt = 0; if(pre00cnt == 2 && byte == "03") continue;
}

��һ�� string ��ʾ�� number, �ж��Ƿ��� "aggregated number"{
	// ���� �ֽ�� һ�����У�ʹ֮���� a[i+2] = a[i] + a[i+1]; like 112358, because 1+1=2, 1+2=3, 2+3=5, 3+5=8 ���Լٶ� a[1] > a[0]
	==> solution: ��ÿһ�������ڶ�����������ʱ��������ȫ�������ˡ�d(i,j) ��ά DP
}

Google Scramble String{
	Scramble string, for two strings, say s1 = ��tiger�� and s2 = ��itreg��
	==> recursive, exponential, ��������ȼ����ĸ����ܿ�
	==> Without Duplicated characters: merge interger interval
	==> solution: logically ajacent {1,0,4,3,2} �� map<char,int> �� vector<int> ����� index; �� vector<int> Left,Right �ϲ�����. 
	ת�� index �Ĵ��룬�� deque push_back ��¼��s1 ���ֵ� index, �� s2 �� deque front pop, ������ map, ��Ϊ��ĸ��һ��
	int n = s1.size();
  if(n != s2.size())return false;
  if(n == 0) return true;  
	vector< deque<unsigned> > Order(26,deque<unsigned>());
	vector<unsigned> Index(n,0);
	int i;
	for(i = 0; i < n; ++i) Order[s2[i]-'a'].push_back(i);
	for(i = 0; i < n; ++i){
	 if(Order[s1[i]-'a'].empty())return false;
	 Index[i] = Order[s1[i]-'a'].front();
	 Order[s1[i]-'a'].pop_front();
	}
	--> ���������� duplicate �� e.g. s1 = "aabbbaccd", s2 = "aabcdbcba", Index = {012683745} Ϊ false, ����һ�ֶ�Ӧ���� Index = {012783645} ��Ϊ true
	==> With "Duplicated" characters:
	==> solution 1 O(n^4): ��̬�滮 p[i,j,k] s1[i:i+k) == s2[j:j+k), �� k = 1...n-i; i, j = 0...n-1; ���뱣���м������Ϊ
		p[i,j,k+1] = || p[i,j,s] && p[i+s,j,k-s] || p[i+s,j,s] && p[i,j+s,k-s]; s = 0,...,k
	==> solution 2: ̫���� http://csjobinterview.wordpress.com/2012/06/29/google-scramble-string-ii-with-duplicated-characters/
	==> solution: DP, O(n). int count[26]; �� s1 ��ͷ��β+1��s2 ��β��ͷɨ��-1, һ������ sum == 0 ��Ϊ�ִ��ϵ�, return true.
}

��n��interval��like this [1,7) [2,4) [5,8) [4,5) [3,6) �ҳ�����Щinterval�ཻ���������ĵ�ļ���{
	Ӧ�÷��أ� [3,4), [4,5), [5,6) ���������Ϸֱ��ص������Σ������ģ�û���ص��Ĵε����䡣
	����[1,7) [2,4) [5,8) ����[2,4),[5,7)
	==> solution: �ؼ�����[,) �Զ˵������Ӱ�졣cnt = 0; ���ж˵�����(�����Ժ�ʡȥ�˶�ÿ�� interval �ļ��)���� [ +1, ) -1�� �ҳ�����ص���. �ڶ��� ͬ��ɨ�裬��¼��������ص����� [ ����һ�� ). Ҳ��ɨ��һ�Σ���¼���� [ , �ڳ��� ) �Ƚ� maximum cnt �� = ������¼���� > ����ռ�¼�ټ���
	==> Extension: �ϲ�����: �� cnt > 0 �����䣬���������� online stream interval. 
	==> Extension, ������ [) ������������ [] () (] ��
	==> solution: �����ж˵�����0(nlgn) ���� n) < n] < [n < (n, ÿ�� n ������ counter: 0 +. cnt ����Ϊ [n: (0+) �� 1; (n: (+) �� 1; n): (0+) �� 1; n]: (+) �� 1.
	==> Extension: ���� ������ص��� ���Խǵ�����ֻ������һά x dimension �� interval, ���� [ , ����Ҫ���������ά���Ƿ�Ҳ�ཻ�����򱣳��ص��ȵ���¼ͬ�ص����µ�������� (���ص����⻹Ӧ��¼ P��)
	==> Extension: ��ά������ά�Կɰ�����չ(�������Խǵ��ÿά����ֽ�)
	==> Extension: ������ص�����(����), �� cnt > 1 ���������(���䳤�ȵĳ˻�)
	==> Extension: ���ص������� [cntLower cntUpper] �ڵ�����ص�����(����), �� cnt �����������������(���䳤�ȵĳ˻�)����
	==> Extension: ���κ�����������������ظ������ص�����: ���� cnt �ڵĸ������������������
}

��һ��records��ÿ��record������������ɣ���ʼʱ�䣬����ʱ�䣬Ȩ�ء��ҵ�һ��set�����set������records��ʱ����û���ص�������set��Ȩ��֮�����{
	==> 1. ������������ �� map, [ +weight, ] ; 2. һ��ɨ�� stack s(1,0); update minInd = 0, Ϊ��ٽ���֮ ], �� [, w(i) + w[minInd] ��ջ. 3. ��ջ�����ֵ����
}

Given array element A[i] represents maximum jump length at i, find the minimum number of jumps from 0 to n-1{
	e.g. A = [2,3,1,1,4], min jump from 0 to 4 = 3
	==> solution: ��󸲸���������. [s,e], cnt; ��ʼ [0,0], cnt = 0; Ȼ�� [e+1,max(A[i]+i,i in ԭ [s,e] )] cnt+= 1; Ϊ �� cnt �����ǵ� interval, ע���ж� end> n.
}

��һ array A[n], s.t. |A_i - A_{i+1}| <= 1; ����һ x{
	==> ��Ҫ��Ϊ�� 2-sorted e.g. 232101, ���Բ����Զ��֡������� ���� |x-A_i| �Ҳ��� O(n). 
}

��ƽ���� n ���� (����)����һ��ʹ֮�������ĵ�{
	==> solution 1: hash б�� space(n^2), ��� O(n^2). one pass
	==> solution 2: ��ÿ���¼ б�ʣ���б���µ�ǰһ�㣬cnt. one pass Ҳ�� O(n^2) space �� time
}

Given ���� A->B->C->D->E->F->G->........->Z, ������ list�е�nodes, C, A, B, E, G �� cluster �ĸ���{
	e.g. Cluster1: A->B->C; Cluster2: E; Cluster3: G. ��������
	==> ע�� node �������ޣ��� list Ԫ������ million
	==> solution: node �� set, ��ÿ��set ��Ԫ�أ����� next �Ƿ��� set �У����� ���� link Ϊ��Ԫ��; ��������Ϊ����. ���Ϊ set �� link Ϊ�����Ԫ�ظ���
}

һ�� strings k-suspious: ������ k-���ȵ� substring ��ͬ, �� a set of strings, �ҳ����� k-suspious substrings{
==> solution: �� hash table (Rabin-Karp rolling hash), ���� string_rolling_hash: val = 0, MUL = 997; for(auto &i:str){ val = (val * MUL + i) % modulers}
--> solution: �� bloom filter ������ substring �ֿ�	
}

�ҳ� 5^1234566789893943�Ĵӵ�λ��ʼ��1000λ����{
	==> solution: ���������� + ���� (1000 λ) �˷�
	
}

һ����������λ 0-9, ��������봮��ʹ�� {// ����: ��ѧ˼ά �� CS ˼ά�Ĳ�ͬ
	==> Hamilton ��· + DFS, ��Ҫ�����ѧ�����ҳ����� // ���� + �ݹ� 
	bool DFS(vector<bool> & IsVisited, vector<char> & Result, int CurrNum){
		if(Result.size() == 10003) return true;
		int pre = (CurrNum % 10000) * 10;
		for(int i = 0; i < 9; ++i){
			int NextNum = pre + i;
			if(IsVisited[NextNum] == true) continue;
			Result.push_back('0'+i);
			IsVisited[NextNum] = true;
			if(DFS(IsVisited,Result,NextNum)) return true;
			Result.pop_back();
			IsVisited[NextNum] = false;
		}	
		return false;
	}
	// initialize
	vector<bool> IsVisited(10000,false); 
	vector<char> Result(4,'0');
	DFS(IsVisited,Result,0);
}

������number��m��n���ҵ�m�� >=n �� ��С Palindromic Numbers{
	==> solution 1: �� n ��ʼһ������֤ m*n
	==> solution 2: ���ɷ� �� n �м�Ľ�λ�� �������� 1. �� n ͬ��; 2. �� n ����. Sub(int len, deque<int>& result, int num = m+1, int start = n)
		--> ע�� n ������ Palindromic, 102 ���� 10 �� 101 ���У������� n �ȳ�ʱҪ�Է��ص� result ɾ������
}

��һ��array of int���Լ�һ��range (low, high), �ҳ����� sum �� range �ڵ������� subsequence{
	==> for L:0 -->n-1; bool IsFromL; 
}
