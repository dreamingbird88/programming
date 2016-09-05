Reading records:
 * 2016.08.21 末完成
 * 2013.10.04
 * 2013.10.24
 * 2013.11.12

Sorting:
 * 与 quick sort 相比，merge sort 的最大特点是，它是一种稳定的排序方法。merge sort 一般多用于外排序。但它在内排方面也占有重要地位，因为它是基于比较的时间复杂度为O(n*lg(n))的排序算法中唯一稳定的排序，所以在需要稳定内排序时通常会选择merge sort。merge sort 不要求对序列可以很快地进行随机访问，所以在链表排序的实现中很受欢迎。
 * 要求常数空间的话，第一考虑是否可用固定数目变量(e.g. 变量的范围)，第二考虑是否可以利用原有空间 (e.g. 原有数组)
 * sorting for duplicates prefer 用 BST + LinkList
 * 可用 indirect sort 来减少 storage and swapping cost, e.g. 用 indirect counting sort 可节省空间


Rolling max (or min) of an array (n) with window(w){// 利用 index queue 来储存信息 Q 长 w, Q[0] is the index of the maximium element within (k-w,k]
	define a de-queue, where the front is the index of maximum before that index,
	since each element enter/remove from that queue at most twice, the complexity is just O(n)
	
	void maxSlidingWindow(int A[], int n, int w, int B[]) {
		if(n <= 0 || n < w) return;
	  deque<int> Q;
	  for (int i = 0; i < n; ++i) {
	    if(i >= w) B[i-w] = A[Q.front()];
	    while (!Q.empty() && A[i] >= A[Q.back()]) Q.pop_back(); // 由于 A[i] 的大小而从后更新, 关键此步
	    while (!Q.empty() && Q.front() <= i-w) Q.pop_front(); // 由于 i 的大小而从前更新
	    Q.push_back(i);
	  }
	  B[n-w] = A[Q.front()];
	}
}

maximum subarray sum {//
	==> solution O(n): 先计算 cumulative sum, O(n); 计算 sum[i] - min_sum[i] O(n)
}

Rolling sum (or mean) of an array (n) with window (w){//two sum 相隔 w, 其差便为 sum with length w.
	void RollingSum(int A[],int n, int w, int B[]){
		if(n <= 0 || n < w) return;
		int i = 0;
		while(++i < n) A[i] += A[i-1]; 
		while(int i = w; i < n; ++i)B[i] = A[i] - A[i-w];
}

Rolling median of online stream{
	==> solution: 维护 min-heap and max-heap, 使其差最多为1, (median 即为多者之root, 若 size 相等，刚为两 roots 之平均值) O(nlgn) Amortized O(lg)
	==> Extension 1: 若求给定 window 的 median? 
			--> Solution: maintain two binary tree, 左树(max heap) 结点数为右树(min heap) 减一，再用one queue 维持过期结点指针。
	==> Extension 2: rolling top kth statistics of online stream 
			* 与 top k 不一样，因为存在元素过期问题，而解决方法是用 queue 记录过期元素在 tree node 里的指针 (需要有 parent 指针，若没有只能像 linkedlist 中删除 cur node 通过 copy 子结点来达到 删除的目的)
			* 与 rolling median 也不一样，因为 median 可以是两数的平均，而不是 windows 内的元素
			--> 解决方法与 rolling median 同理，维持 two tree, 一边是 min tree 存 top K 元素，一边是 max tree 存 windows 内的其他元素 (这些数得留着做更新)。 而用 queue<node*> 负责更新 index. 求所有rolling Top K 元素 complexity 为 O(nlgk)
	==> Extension 3: 在一个不知长度的数组里找 第 k th element, 要求 O(n) time, O(k) space;
			--> solution 1: min-heap O(n lgk); solution 2 (妙): Garbage collection 机制: 维护 2k-1 数组，每到 2k-1 时 更新一次，留下 top k 个
}

维护不定元素的 Min-Stack{
==> Min-Stack, with pop(), push() and min() in O(1) // 用 rolling min Q
		Solution 1: double the space, Using additional stack to store CurrentMin.
		Solution 2: O(n) computation in pop() if the popped element is the minimum element
		Solution 3: maintain an auxilary min-index stack, Q[end] = current mininum element index; Q[k-1] = the index of the minimum element after Q[k] is removed, 所以 Q 里的元素肯定是非增的
}

维护不定元素的 Min-Queue{ // with pop(), push() and min() in O(1)
		Solution 1: double the space, Using additional stack to store CurrentMin.
		Solution 2: 用一个 index queue 及 timestamp, 入元素时：当 q.back() > A[i], q.push_back(t); 出元素时: 当 timestampe - q.front() < Queue size 时，pop()
}

给定 n 个数，求 x, 使得 \sum_{n} |v_n - x| 最小，即求中位数{
	==> Extension: 有一栋楼，一共有N层，要去每一层的人分别是A[1],A[2]....A[N]，如果电梯可以停K次，问停在哪K层让所有人走的矩离最短
	==> Solution: opt_k[j]--电梯前k次停在1到j层之间 1-j 的人的最优解, cost[i][j]--第i层到第j层停一次让 [i,j]之间的人走最优解
		opt_{k+1}[i]=min_t{opt_k[t]+cost[t+1,i];} (i<=n) 其中 opt_1[k] = c[1][k], 而 c[i,j] = median(i,j) 用 rolling median 求解只需 O(n^2)
		而求动态递归只需 O(k*n*n), 所以共需 O(k*n^2)
}

给集合 S = {1,...,n}, 及两等长 index array A[K], B[K] with  {A[k],B[k]} 在同一集合. 求 A[K], B[K] 把 1,...,n 分成的所有子集{
	==> 思路: 建一个 I[n], 初始化为 -1; I[i] 为 元素 i 对应的 list 首指针(最好还有尾指针，方便合并); 每遇一元素 A[k], 若 I[A[k]] 为空，则建新指针; 若非空且两个 list 不同，则将新 list 接到原list 上，且更新新 list 上各元素的 I[i];
}

Given an array of numbers, nums, return an array of numbers products, where products[i] is the product of all nums[j], j != i.{
	Input : [1, 2, 3, 4, 5]
	Output: [(2*3*4*5), (1*3*4*5), (1*2*4*5), (1*2*3*5), (1*2*3*4)] = [120, 60, 40, 30, 24]
	You must do this in O(N) without using division. 
	==> solution: 两边计算 Prefix and Suffix
}

Find the Minimum Non-Existing Positive Integer{ // 利用 array index 的信息来判断缺少元素或是重复元素
[3,4,-1,1] returns 2 // 注意 4 与 1 交换时，1 还需要继续交换

Can you do it in O(n) time and O(1) space?
==> Solution: swap n to a[n], then find the first k, a[k] != k;
or Find all duplicates in an array of size N where the values in the array can be from 0-N

int MinNonExistPosInt(int * Array, int Len){
	for(int i = 0; i < Len; ++i)	{
		while(Array[i] > 0 && Array[i] <= Len)		{
			if(Array[i] != Array[Array[i]-1]){
				int temp = Array[Array[i]-1]; //正确: 交换过程中下标已经改变
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
	==> 把问题深化为求 第 k 个元素 比较好写 code.
    double findKthElement(int A[], int m, int B[], int n, int k) {// m + n >= k
        if(m == 0) return B[k-1];
        if(n == 0) return A[k-1];
        if(k == 1) return min(A[0],B[0]);
        int a = min((k-1)/2,m-1);
        int b = min(k-2-a,n-1); // a + b <= k - 2 容易写错的地方
        if(A[a] == B[b]) return B[a];
        // if A[a] < B[b];  for A[i], i <= a, at most i+b <= a+b <= k - 2 element <= A[i], therefore eleminate A[i] up to i == a
        return A[a] < B[b]? findKthElement(A+a+1,m-a-1,B,n,k-a-1): findKthElement(A,m,B+b+1,n-b-1,k-b-1);
    }
}

Given two sorted arrays A, B of size m and n respectively. Find the k-th smallest element in the union of A and B{

You can assume that there are no duplicate elements.

==> solution: O(lg(min(m,n,k))), or lg(k) 每次去除 k/2 个 (至少一个 array 里的前 k/2个)，二分法 
	int KthElementTwoArray(int * A, int a, int * B, int b,k){
		if(a == 0 && b == 0 || k > a + b) throw error;
		if(a == 0) return b[k-1];
		if(b == 0) return a[k-1];
		int ai = min(k/2,a-1), bi = min(k/2,b-1);
		return a[ai] < b[bi]? KthElementTwoArray(A+ai+1,a-ai-1,B,b,k-ai-1):KthElementTwoArray(A,a,B+bi+1,b-bi-1,k-bi-1);
	}
	
==> Extension:
	(1) Given a N*N Matrix. All rows are sorted, and all columns are sorted. Find the Kth Largest element of the matrix.(Young Matrix?)
			==> solution: 计算出kth元素所在的反对角线，再在该反对角线内查找 O(min(k,n))
	(2) what about M arrays? 
			==> solution 1: O(klgk) using k-heap, each element needs lg(k) // too bad
			==> solution 2: 与两组一样，不过每次只能去 M/k * 2 个元素
}

2D Young 氏 matrix 查找 是否存在某数{
	==> 每次比较 A[0][n-1]; 由此确定 删除 列还是行
}


长为n的数组里找离median最近的k个数，要O(n)的complexity{//Find unsorted 数组前 k 个数只需要 O(n);
	==> Solution 1: O(nlgn) 先 sort then O(k);
	==> Solution 2: O(n+k) (a) Find median with Quick sort O(n) amortized: 先 RANDOMLY select pivot value, 再 split array; (b) 再将 2 * k 扫描一遍即可
}

给 partially sorted array A[n], 每个元素偏离正确位置 at most k, sort 这个数组 O(nlgk) in space{
	==> 用 k length min-heap, 每次移位一个，再调整 heap O(lgk)
}

10.10 判断 heap 中第k元素与 x 的关系, 要求时间与 space 皆在 O(k) {// ???
	==> 用 EquCnt, LagCnt遍历 O(k) 
}

KMP快速字符串查找算法(Knuth-Morris-Pratt) search a word w in a string s, O(w+s) instead of O(w*s) {
==> Build an array for length of w: I[0] = -1; where I[i] meaning index of the elment in W should be compared when the ith element is not equal. e.g

修正 a b a b 不是 -1 0 0 1, 而应是 -1 0 -1 0, 多一个 

int * get_KMP_Ind(char*str, int Len){
	if(Len == 0) return 0;
	if(Len == 1) return new int(-1);
	int * I = new int(Len);
	I[0] = -1; 
	int i = 0, j = -1;
	while(i < L-1){// compare str[0:j] with str[i-j:i], 有 ++i 操作，所以 i < L-1 而不是 i < L
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

BM Boyer-Moore 算法{// 将 String Pattern 左对齐，从 P 最后字符开始比较。 strsub 实际使用的
	http://blog.csdn.net/v_JULY_v/article/details/6545192
	http://www.inf.fh-flensburg.de/lang/algorithmen/pattern/bmen.htm
	http://www.ruanyifeng.com/blog/2013/05/boyer-moore_string_search_algorithm.html
	BM Boyer-Moore 算法 与 KMP 算法类似，KMP 是最好前缀组，而 BM 是最好后缀组，不能用递归，比较复杂 O(n)
	BM算法就是找 坏字符规则 最好后缀规则 中的最大位移, 当有一个 mismatch 时，计算 bad character 和 good suffix 的位移
	1) Bad character heuristics： 一个字母表 index，若pattern 中没有，则长度为 Pattern_length, 否则，为pattern 里从后到前最近的位置
	2) Good suffix heuristics： 
}

Given a string S, find the longest palindromic substring in S{
	==> Solution 1, O(n^2): check each element as center of possible substring, need to consider "abba", not just "aba", 插入 空格
	==> Solution 2, O(n^2): DP; p[i,j] = true if i...j is palindromic. Initial p[i,i] = true, p[i,i+1] = true is s[i] == s[i+1]; p[i-1,j+1] = p[i,j] && s[i-1] == s[j+1];
	==> Solution 3, O(n): Let Len[i] be the length of palindromic with center on i; if c is the index of current center and d <= Len[c], then Len[c-d] <= Len[c+d]; when 以 c-d 为中心的子串在 以 c 为中心的串内时(即 c-Len[c] < c-d - Len[c-d]) 或者 Len[c+d] 到末尾时, 等号成立. 若等号不成立，则可以 c+d 为中心(注意 d 从 1 到 Len[c]) 
	==> Solution 4, O(n): Suffix trees. S and S', if the LP Substring center at c, then the center in S' is at n-c+1. check whether (c,n-c+1) is the 
			http://blog.csdn.net/linulysses/article/details/5634104
}

将一字串中 'b' 删除 及 'a' 变成 'aa' in O(1) space, 假设原串有足够空间变换后的新串{
	==> solution 2: 从尾开始 replace. 
	==> solution 1: 先扫一遍根据 'b', 'a'	个数计算出新串长度，再从尾开始
}

Search in Rotated Sorted Array{
	Suppose a sorted array is rotated at some pivot unknown to you before hand.
	
	(i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
	
	You are given a target value to search. If found in the array return its index, otherwise return -1.
	
	==> If no duplicate exists in the array: O(lgn), 根据 A[s] A[c] A[e] 的关系来判断所在那一段
	==> If duplicate exists in the array: 在于三者相等时，难以判断左右, 其他情况皆可分段
}

Given a array of integers , find 3 indexes i,j,k such that, i< j < k and a[i] < a[j] < a[k] {// ???
	==> Solution: O(lg n), two candidate index set, with s2[0] < s1[0] < s1[1], for any i, if(a[i] > s1[1]){found!}else{if(a[i] > s2[0]){replace s1 with s2[0] and a[i], clear s2;}
	}
}

一个数组中任意一个local min(比它左右都小)就可以{// ???
	比如1,2,3,7,19,8,6,11,34,23,67 里面的6比8,11小，23比34,67小，都是 local min。
	==> solution O(lgN) 不重复的数: 若 b > m < e, if(m-1 < m) search [b,m] else search [m-1,e]
}

在一个int数组里查找这样的数，它大于等于左侧所有数，小于等于右侧所有数{// 两端的数算什么?
	==> solution: 从左到右扫描，将 max index 入 stack; 从右到左扫描，遇到 min 则与 stack 里的 max 比较是否同一 index
}

让你排序N个比N^7小的数，要求的算法是O(n){// 给了提示..说往N进制那方面想，bucket sort, 注意 7 是常量
	==> solution: 两头扫描，将比 N^6 大的数移到最左; 再次就是 N^5, N^4, ... N, 一轮就是 O(n); 记下分段地址，再下一轮, 共7轮
}

求随机数构成的集合中找到长度大于=3的最长的等差数列{// ???
	eg. 输入[1,3,0,5,-1,6], 输出[-1,1,3,5]. 要求时间复杂度，空间复杂度尽量小
}

count the number of reverse order pair{
	Give an integer array A[] with size of n, count the number of reverse order pair.
	e.g. [6 2 5 3 9], pairs: (6,5), (6,3), (5,3) count = 3;
	==> Solution: insertion sort/bubble sort + count O(n^2)
	==> Extension: give B[] and A[], count the number of reverse order of A[]'s elements in B[]'s order
}

find the kth integer that contains only factors 2,3,5, starting from 1 {
	==> solution 1 : O(k), 3 queue, each for 2,3,5. find the minimum of the three, if it pop up from the small number, then multiple with number that greater than it // 注意溢出
		
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
	==> solution 2: [1,k*2] 间找一整数 i，使得 f(i) := i/2 + i/3 + i/5 - i/6 - i/15 - i/10 + i/30 < k but f(i+1) >= k?
}

Maximum profit from stocks {
	http://www.leetcode.com/groups/google-interview/forum/topic/maximum-profit-from-stocks/
	Given M different stocks prices across N different day and an Amount D dollars. how do you maximise your profit by buying and selling stocks. more formally: this is an MxN matric. Each row corresponds to a stock and N columns correspond to price of the stock on day “i”. You are given D dollars to buy and sell stocks. maximize your profits.
	==> If can buy and sell daily: for each stock, calculate p_{i+1}/p_i to get a M*N matrix; then for each day, find the max(d)
	==> If one can only buy and sell once: for each stock, find the p_max/p_min; find the stock with the max p_max/p_min
		Best Time to Buy and Sell Stock{// 只能一次交易，找之前最低点
		    int maxProfit(vector<int> &prices) {// 2013.09.23 无错通过
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
		Best Time to Buy and Sell Stock II{// 可以多次交易，但最多同时有一个股票
		    int maxProfit(vector<int> &prices) {// 2013.09.23 无错通过
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
		
		Best Time to Buy and Sell Stock III{// 可以最多交易两次，不能同时进行
			==> 从左到右 for each i, 计算第一次以 i 结束的交易 maximum profit; 从右到左 for each i, 计算第二次交易在 (i,e] 内发生的 profit
			// Error 1: 逆向计算时沿用 PreMin, 应为 PreMax, 而且 PreMax 应在前
		    int maxProfit(vector<int> &prices) {// 2013.09.23 用动态规划，分两个子串，前后扫描
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
	==> solution 1: sort O(nlgn) and find O(n) with space complexity O(1), 两头扫描
	==> solution 2: hash(a_i) and hash(sum - a_i) with time complexity O(n) but space complexity O(n);
	==> Extension: 在 abs-sorted 的 array 里找 two sum: two pointer，一个正数，一个负数
	==> Extension: 两数差 为 k 
}

Three Sum, without extra space{
		* Solution 1: O(n^{k-1}), similar to K-sum: 即 hash table, 然后查 -(a[i] + a[j])的 value; 
		* Soultion 2: Using extra space, C_n^{k/2} and O(n^{k/2} lgn) if k is even; C_n^{(k+1)/2} and O(n^{(k+1)/2}) if k is odd. Using extra space 不解
		* Solution 3: O(n^2): 先 sort O(nlgn), 然后每个数，看能否找到另两个使三者之为 0 O(n^2)
		* Conclusion: for K-sum, requires O(n^{Omega(K)}) <On the Possibility of Faster SAT Algorithms> by Mihai Patrascu and Ryan Williams.
}

N sum problem{
		* solution 1: recursive enumerate
		* solution 2: backpack problem 
		* extension : 有两个序列a,b，大小都为n,序列元素的值任意整数，无序；要求：通过交换a,b中的元素，使序列a元素的和与序列b元素的和之间的差最小。
}

两个string, 给出它们的两个等长的substring, 定义它们的距离为distance=sum_i(s1[i]-s2[i])，找距离最大的两个substring{// DP 
	==> If without absolute "s1[i]-s2[i]"
	==> If with absolute "abs(s1[i]-s2[i])"
	==> Solution DP or 穷举: O(n*m) time + O(n) space
	f(str1,str2) = max{f(str1-1,str2), f(str1,str2-1), g(str1-1,str2-1)+ str1[end] - str2[end]},where g(str1,str2) is the max distance of suffix. 
}

一个单词的列表，要找出两个单词，它们没有相同的字母出现，而且长度乘积最大{
	==> Solution: 根据单词列表构造一个 26 层的树. 第一层 'A' yes or no, Node{int MaxLen} 按照每个 word 的 signature, bool IsInWord[26]; 只存符合该 signature 单词最长的长度。中间节点的 MaxLen 表示其子节点中的最长长度。 然后找 complementary 总共 O(n)
}

找 string s 中最长中 lexicographic order 最大的无重复字母的子串{ eg: Input "babab"; Output "ba"
==> solution: 计算 s signature bool IsInWord[26], 这样就得出所求 substring len = sum(IsInWord). 
}
Longest Substring Without Repeating Characters {
	找 A[] 中最长短子串，使之不包含重复字母：用 Begin[26], 及 startInd, 只有在 Begin[26] > startInd 才更新 MaxLength.
	==> 思路: 用 C[26] 记录最迟出现的 index, Pre 不重复字串首字符前的坐标，即不包括 A[Pre]
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


找minimum window in A, contains string B {// 注意 是有次序的
input
A = [1, 9, 3, 4, 12, 13, 9, 12, 21]
B = [9, 12, 21]

output
A[6..8] = [9, 12, 21]

建立一个 B.length() 长的数组，记录下一个元素是 B[i] 的 window 在 A 中 最开始元素位置. 再建一个 vector<int> Set[26], Set[i] 对应 'a'+i 在 B 中出现的元素位置
==> 每个解都由 BeginInd, 及 NextElementInB 构成
==> One-pass solution: maintain a queue Q with element <BeginInd,NextElementInB>, initially, it has an element <-1,0> 
	scan A one-pass, for each element A[CurInd], update Q with
	  1. if A[CurInd] == B[NextElementInB]; if (NextElementInB == 0){ BeginInd = CurInd}; update NextElementInB += 1; 
	  2. Delete q[b1,n1] if existing q[b2,n2] in Q, s.t. b1 <= b2 and n1 <= n2;
}

Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n){
	// 与上题相比是 字母集很小，而整数集很大, 且这里次序无关
	==> solution: 1. 用 cnt[26/52] 统计出 T 出现的字符数，及字符种类个数 posCnt, negCnt = 0; 2. right=0 <n 更新cnt posCnt negCnt 使 posCnt = 0; left = 0 < right 更新cnt posCnt negCnt 使 negCnt = 0; 3. 若 posCnt = negCnt = 0 表明找到包括 T 的 S 子串, 更新 left 到第一个 T 的字符，则找到此段最短子串, 更新 Max[2](起始与长度)。若只需包括 T, 则只需要维持 posCnt == 0
	For example,
	S = "ADOBECODEBANC"
	T = "ABC"
}

compute the lexicographically smallest permutation of [1,2,....n] given a signature{//
	signature: eg {3,2,1,6,7,4,5} ==> signature = "DDIIDI" --> 最小 permutaion = 3,2,1,4,6,5,7;
	==> solution O(n): 1. from "DDIIDI" get I[7] = [2,1,0,0,1,0,0] 从尾到头遇 'D' 加 1，遇 'I' 归 0. 
		2. 从 Min = 1; 开始每次遇到 'D' 则 nextMin = I[i] + Min + 1; I[i] = Min + I[i]; 直到计算完第一个 I[i] == 0 后，reset Min = nextMin; 遇到 'I' 则 I[i] = Min; Min += 1;
}


Given an int array A[n], higher value more candy than neighbor, every A[i] at least got a candy, find the minimum number of candy{
	==> c[i] 为 i 的糖果数，则 c[i] 是 min{ > i 连续递减到 i 最长串的数目 ，< i 的元素里连续递增到 i 最长串的数目}
	可以分两遍扫；c[] 初始化为 1
1. 第一遍从前向后扫严格单调增区间，c[i] = c[i-1]+1;
2. 第二遍从后向前扫严格单调减区间，c[i] = max(c[i],c[i+1]+1);其中c[i]是第一遍扫单增区间的赋值，c[i+1]+1是本次扫单减区间的赋值；似乎只用考虑上拐点就可以了，
}


Rotation an array A[n] k position with O(1) space and O(n) time{
	==> 简单的解法 reverse the whole A[1-n], then reverse A[1-i], A[i+1, -- n] separately
}

Largest Rectangle in Histogram{
	==> O(n) with stack; stack 比前高的元素 height 及 index ，遇到低的则统计之前的. 为此应在 input 里加最后元素 0
}

Trapping Rain Water {
	==> 与 Largest Rectangle in Histogram 相对应，易犯错误: 1. 找左边开始 bar; 2. 若右 bar 比左bar 高如何办? 维持 stack.front() 为目前最高 bar
}

Container With Most Water{// 想复杂了, 应该是 
	Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.
==> Solution: i= 0, j = n - 1; 那么当前的area是min(height[i],height[j]) * (j-i); while (i < j){ if(height[i] < height[j])++i; else --j}
}

Maximal Rectangle，找 0,1 矩阵里 1 构成的最大子矩阵{
	==> solution 1 O(n^3): p(i,j) 该点开始最大矩阵，每行保持每点开始的最大距离, 试 k >= i; m = min(l_k,l_{k+1}}, 当 m  为0 或 k = n-1 时停
	==> solution 2 O(n^2): 利用 Largest Rectangle in Histogram， 对每行找以该行为底的最大子矩阵 r = 0-->n-1，每行 O(n). 柱子高度及位置更新在循环内. 注意在更换柱子时，新柱子的 index 要改成pop 出的柱子的
}


Maximun sub matrix n*n (2d version of maximum subarray){// 连在一起的
	==> 思想: 1. 避免重复计算; 2. 将高维化成低维
	==> Blue force O(n6)
	==> solution O(n3): 1. for each column, calculate prefix sum O(n^2); 2. for each pair of rows (rs,re) and each column c calculate f(rs,re,c) as the sum from rs to re in column c. O(n^3); 3. For each (rs,re,*), find the maximum subarray. O(n^3)
	==> extension: d-dimension matrix, maximum submatrix O(n^{2*d-1})
}

你共有m个储存空间。有n个请求，第i个请求计算时需要 R[i]个空间，储存结果需要O[i]个空间(其中O[i]<R[i]), 求安排这n个请求的顺序，使得所有请求都能完成{
	e.g. 你的算法也应该能够判断出无论如何都不能处理完的情况。比方说，m=14，n=2，R[1]=10，O[1]=5，R[2]=8，O[2]=6
==> solution 考虑最简单情况(两个工作) i,j. If run i first: O_i + R_j; if run j first: O_j + R_i; then if O_i + R_j < O_j + R_i, run i first. i.e. run max(R-O) first. 
}

Having an [infinite] stream of numbers write a function to take an element with equal probability for each{
	虽然是无限，但是我们得保证前 N 个元素每个概率是 1/N, 其实只要保证第 N 个元素(最新)会有 1/N 概率被选。具体做法是：  
	保持一个以前元素，及数组长度 N，每次 1-1/N 的机会 新数 代替 前数，当 stream 停止时，该数就是所选的 number  
}

有2*N个文件，文件的大小保存在size[2*N]中。然后想要分成N份(每一份可以有1或者多个文件)，要使这N份中的文件size之和的最大值最小{
	==> 近似解: sort一遍 从大往小排n个 然后剩下n个，从大往小，尽量塞到高度最小的文件中去
}

两个排序好的数组  求和最小的m个pair{//
	1) insert A[0] B[0]
	2) pop up the smallest, insert A[1]B[0],A[0]B[1]
	3) pop up the smallestA[i]B[j] , insert A[i+1]B[j],A[i]B[j+1]... 	until u get m
	==>  有点像 Young matrix，反对角线进行 O(m)
}

Given an array A of N integers, where A[i] is the radio at i, find the number of intersecting pairs{
	e.g. A = {1,5,2,1,4,0} 共有 11 对
	==> solution: time O(n) + space O(min(n,radio)). 第一遍扫描 c[i] 表示 左边到达 i 的个数 (不包括 i 本身). 第二遍 sum up: m(i) := min(i,A[i]); sum += m(i) + c[A[i]-m(i)] 
}

Given an unsorted array of integers, find the length of the longest consecutive elements sequence.{
	==> O(n): 1. 所有元素入 unordered_set, 2.任取一元素 v, 按 v+1 up, v-1 down 在 set 中查找，遇到所查元素 count 1 并消去 
}

将书组里的所有负数排在所有正数前面，保持正数和负数原来的相对顺序不变 inplace 时间复杂度越低越好{// ???
	eg: input -5 2 -3 4 -8 -9 1 3 -10;  output -5 -3 -8 -9 -10 2 4 1 3.
	==> 方法类似于把一篇文章以单词为单位头尾调换. nlogn
}

Palindrome Partitioning{// given a string, 分解成 若干 palindrome strings. (1) 求出所有可能的组合 (2) 找出 substring 数目最少的切割 2013.09.13
	--> 思路 DP: 构造二维 bool p(i,j), substring[i,j) is palindrome. 递归 p(i,j) = p(i+1,j-1) && a[i] == a[j-1]; 从矩阵左下角开始; 边界 p(i,j) = true if i+1 >= j; 
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

Reverse Nodes in k-Group{// 2013.09.23 VC 通过，但无法通过 LeetCode
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

Remove Element that equal to a given value{ // LeetCode 不能通过 [2,2,3], 2, 但 VC 调试没有问题
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
    int removeDuplicates(int A[], int n) {// 2013.09.23 无错通过
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

Remove Duplicates from Sorted List{// 2013.05.02 8:06pm 8:09pm 无错通过
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

Remove Duplicates from Sorted List II{// 2013.05.02 8:10pm VC 调错 8:37pm 
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
                // 出错: pre = cur; 多此一举
            }
        }
        return tempN.next;        
    }	
}
    
Reverse Linked List II{// 2013.05.02
    ListNode *reverseBetween(ListNode *head, int m, int n) {// 需要 VC 找错误
        if(m >= n || !head) return head;
        ListNode tempNode(0);
        ListNode* Pre1st = &tempNode;
        n = n-m; // number of reverse
        
        Pre1st->next = head; // 关键生成一个头来处理 head reverse 的问题
            
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
        return tempNode.next;// 错误 Pre1st->next;忘记 Pre1st 可能已经改变
    }
    
    // 2013.08.04 出错，变成交换 n-m 次, 都忘记了，唉
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

Reverse Nodes in k-Group{// 不够的不用交换 // 无错通过! 2013.05.02
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

Remove Nth Node From End of List{// 无错通过 2013.05.02, 但 2013.09.21 有错
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

Partition List{// VC 查错 2013.05.02
// 错误一: 交换 linked list 内两点，相邻时用 pre->next->next 会有麻烦
// 错误二: 算法设计错误, 不是交换两点，而是改变四个链接

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

Merge Two Sorted Lists{// 出错 && 写成 & 2013.05.02 // 判断指针是否为 null 时取反了, e.g while(!p){}; 应为 while(p){}
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

Merge k Sorted Lists{// 用 VC 调出错误 2013.05.02
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
                MinHeap[0] = MinHeap[size-1];// 出错: MinHeap.pop_front(); 改变 size 后没有重排，有关 MinHeap 出错
                MinHeap.pop_back();
                --size;
            }
            RearrangeMinHeap(MinHeap, 0);
            pre = pre->next;
        }
        pre->next = (size == 0? NULL:MinHeap[0]);// 出错 为 size==1
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

Rotate List{//没有完成 2013.05.02 10:42pm
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
	==> 注意对 "*" 理解错误, 正确是表示对"前面"字符的任意次(包括 0)的重复
	==> Do NOT working: 用 DP, 即 str 与 pattern 的最长公共字串, 不要用正向 recursive, 因为 complexity 为 exponential. pattern 必须 表达唯一, eg. b*b 不合法，而 bb* 合法
	
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

匹配人名 通配符 * and ?; e.g. “J* Smi??” 可以匹配“John Smith”{
==> solution: 分开 *, ? , 三种情况 讨论 how to handle J*ss Jesass, 

bool IsFit(const char* txt, int txtSize, const char* pat, int patSize){// recursive, 错误 const char* 不能赋值给　char*
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
	
	if(txtSize < shift || patPos == 0) return false; // not enough txt or pat[patPos] != '*','?'; 错误 forget "|| patPos == 0"
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
==> Solution 1: Knuth–Morris–Pratt algorithm(KMP): build index for search pattern, p[i] is that if i th charactor fails, which index should go to 
==> Solution 2: Trie tree: prefix tree for suffix, good for large number of small patterns, 
==> Solution 3: Rabin–Karp hash: build hash functions (rolling hash, fixed lengths as the searched patterns) for patters and each substring of the text. good for search multiple patterns. e.g f(str) = str[0]+str[1]*p+...+ str[k-1]*p^(k-1), where p is a large prime
}

copy LinkedList{ where m_pSibling 指向链表中的任一结点或者NULL。其结点的C++定义如下：
	 struct ComplexNode
	{
	    int m_nValue;
	    ComplexNode* m_pNext;
	    ComplexNode* m_pSibling;
	};

	==> Solution: copy each oldnode (recursively, then next can be copied to new node), with the newnode->next = oldnode, and oldnode->next = newnode; 2. update noldnode->next->sibling = oldnode->next->next 即新点的 sibling update, 3. 新点从旧点中解脱出来 
}

reverse Polish Expression{
	1 + 2 * 3 = 2 3 * 1 +;
	==> 用辅助 stack 入数，每读到一个运算符从 stack 中取得两数计算
}

链表问题的面试题目{
1.给定单链表，检测是否有环
	==> Solution: 使用两个指针p1,p2从链表头开始遍历，p1每次前进一步，p2每次前进两步。如果p2到达链表尾部，说明无环，否则p1、p2必然会在某个时刻相遇(p1==p2)，从而检测到链表中有环。

2.给定两个单链表(head1, head2)，检测两个链表是否有交点，如果有返回第一个交点, 注意 L1, L2 可能有 circle. 
	==> Solution: 如果head1==head2，那么显然相交，直接返回head1。否则，分别从head1,head2开始遍历两个链表获得其长度len1与len2，假设len1>=len2，那么指针p1由head1开始向后移动len1-len2步，指针p2=head2，下面p1、p2每次向后前进一步并比较p1p2是否相等，如果相等即返回该结点，否则说明两个链表没有交点。


3.给定单链表(head)，如果有环的话请返回从头结点进入环的第一个节点。
	==> 我们可以检查链表中是否有环。如果有环，那么p1p2重合点p必然在环中。从p点断开环，方法为：p1=p, p2=p->next, p->next=NULL。此时，原单链表可以看作两条单链表，一条从head开始，另一条从p2开始，于是运用题二的方法，我们找到它们的第一个交点即为所求。

3.2 判断一个 linkedlist 是否是 palindrome

3.3 把一个 linkedlist 对折地相隔变换成一个新 linkedlist

// 以下两题可以将旧节点转化成新节点
4.只给定单链表中某个结点p(并非最后一个结点，即p->next!=NULL)指针，删除该结点。
	==> 首先是放p中数据,然后将p->next的数据copy入p中，接下来删除p->next即可。

5.只给定单链表中某个结点p(非空结点)，在p前面插入一个结点。
  ==> 办法与前者类似，首先分配一个结点q，将q插入在p后，接下来将p中的数据copy入q中，然后再将要插入的数据记录在p中。	
}

// 以下用到字符串的 signature
Given a string S and a list of words (same length) L. Find all starting indices of substring(s) in S that is a concatenation of each word in L exactly once and without any intervening characters{
	==> solution: O(n) 1. 用 unordered_set 使查询是否在 list 里为 O(1); 2. DP 维持 word.size() 个 deque<bool> queue，Q[i] 表示从 i 开始已经出现的 words, 还有 vector<bool> Appear(word.size(),false)
}

Substring with Concatenation of All Words{ 
     vector<int> findSubstring(string S, vector<string> &L) {// L is not empty
        // 不能用于 S:"a"; L: ["a","a"], 需要用到 unordered_multiset
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
	unsigned char read_byte(); ← side effect that it advances a byte pointer in the stream
	write:
	unsigned char read_sync_byte(); ← may result in >1 calls to read_byte()
	
	remove byte '03' from the stream if the stream is in pattern 00 00 03
	read_byte():
	00 0f 42 17 00 00 03 74 00 00 00 00 14 ...
	read_sync_byte():
	00 0f 42 17 00 00 74 00 00 00 00 14
	
	==> solution: 用 ring buffer, Buffer[0-5], start, len; if str[cur] == pattern[len], ++len; else read out; if len == pattern,
	==> solution: int pre00cnt = 0; if( byte == '00'){pre00cnt +=1; if(pre00cnt == 3) pre00cnt = 2;} else pre00cnt = 0; if(pre00cnt == 2 && byte == "03") continue;
}

给一个 string 表示的 number, 判断是否是 "aggregated number"{
	// 定义 分解成 一个数列，使之满足 a[i+2] = a[i] + a[i+1]; like 112358, because 1+1=2, 1+2=3, 2+3=5, 3+5=8 可以假定 a[1] > a[0]
	==> solution: 当每一个数及第二个数定下来时，数列完全定下来了。d(i,j) 两维 DP
}

Google Scramble String{
	Scramble string, for two strings, say s1 = “tiger” and s2 = “itreg”
	==> recursive, exponential, 但是如果先检查字母数会很快
	==> Without Duplicated characters: merge interger interval
	==> solution: logically ajacent {1,0,4,3,2} 用 map<char,int> 与 vector<int> 找相对 index; 用 vector<int> Left,Right 合并区间. 
	转换 index 的代码，用 deque push_back 记录在s1 出现的 index, 对 s2 用 deque front pop, 不用用 map, 因为字母数一样
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
	--> 不可用于有 duplicate 的 e.g. s1 = "aabbbaccd", s2 = "aabcdbcba", Index = {012683745} 为 false, 但另一种对应排列 Index = {012783645} 则为 true
	==> With "Duplicated" characters:
	==> solution 1 O(n^4): 动态规划 p[i,j,k] s1[i:i+k) == s2[j:j+k), 从 k = 1...n-i; i, j = 0...n-1; 必须保存中间过程因为
		p[i,j,k+1] = || p[i,j,s] && p[i+s,j,k-s] || p[i+s,j,s] && p[i,j+s,k-s]; s = 0,...,k
	==> solution 2: 太复杂 http://csjobinterview.wordpress.com/2012/06/29/google-scramble-string-ii-with-duplicated-characters/
	==> solution: DP, O(n). int count[26]; 对 s1 从头到尾+1，s2 从尾到头扫描-1, 一旦出现 sum == 0 即为字串断点, return true.
}

有n个interval，like this [1,7) [2,4) [5,8) [4,5) [3,6) 找出与这些interval相交的最多次数的点的集合{
	应该返回， [3,4), [4,5), [5,6) 这三个集合分别重叠了三次，是最多的，没有重叠四次的区间。
	给定[1,7) [2,4) [5,8) 返回[2,4),[5,7)
	==> solution: 关键在于[,) 对端点记数的影响。cnt = 0; 所有端点排序(排序以后省去了对每个 interval 的检查)，遇 [ +1, ) -1， 找出最大重叠数. 第二遍 同样扫描，记录出现最大重叠数的 [ 到第一个 ). 也可扫描一次，记录最后的 [ , 在出现 ) 比较 maximum cnt 若 = 则加入记录，若 > 则清空记录再加入
	==> Extension: 合并区间: 即 cnt > 0 的区间，但不能用于 online stream interval. 
	==> Extension, 若不是 [) 而包括有其他 [] () (] 呢
	==> solution: 将所有端点排序，0(nlgn) 其中 n) < n] < [n < (n, 每个 n 用两个 counter: 0 +. cnt 规则为 [n: (0+) 加 1; (n: (+) 加 1; n): (0+) 减 1; n]: (+) 减 1.
	==> Extension: 矩形 的最大重叠： 两对角点坐标只对其中一维 x dimension 用 interval, 若有 [ , 则需要检查在其他维度是否也相交，否则保持重叠度但记录同重叠度下的新入点坐 (除重叠度外还应记录 P点)
	==> Extension: 三维甚至多维皆可按此扩展(即两个对角点的每维坐标分解)
	==> Extension: 求最大重叠区域(区间), 找 cnt > 1 的区域面积(区间长度的乘积)
	==> Extension: 求重叠次数在 [cntLower cntUpper] 内的最大重叠区域(区间), 找 cnt 符合条件的区域面积(区间长度的乘积)即可
	==> Extension: 矩形合起来的总面积，不重复计算重叠部分: 计算 cnt 内的各个面积，用总面积相减
}

有一组records，每个record由三个参数组成，开始时间，结束时间，权重。找到一个set，这个set包含的records在时间上没有重叠，并且set的权重之和最大{
	==> 1. 区间两端排序 用 map, [ +weight, ] ; 2. 一次扫描 stack s(1,0); update minInd = 0, 为最迟结束之 ], 遇 [, w(i) + w[minInd] 入栈. 3. 求栈中最大值即可
}

Given array element A[i] represents maximum jump length at i, find the minimum number of jumps from 0 to n-1{
	e.g. A = [2,3,1,1,4], min jump from 0 to 4 = 3
	==> solution: 最大覆盖区间问题. [s,e], cnt; 开始 [0,0], cnt = 0; 然后 [e+1,max(A[i]+i,i in 原 [s,e] )] cnt+= 1; 为 第 cnt 跳覆盖的 interval, 注意判断 end> n.
}

给一 array A[n], s.t. |A_i - A_{i+1}| <= 1; 查找一 x{
	==> 不要认为是 2-sorted e.g. 232101, 所以不可以二分。但可以 根据 |x-A_i| 找步长 O(n). 
}

给平面上 n 个点 (整数)，画一线使之包含最多的点{
	==> solution 1: hash 斜率 space(n^2), 最多 O(n^2). one pass
	==> solution 2: 对每点记录 斜率，该斜率下的前一点，cnt. one pass 也是 O(n^2) space 与 time
}

Given 无序 A->B->C->D->E->F->G->........->Z, 给几个 list中的nodes, C, A, B, E, G 求 cluster 的个数{
	e.g. Cluster1: A->B->C; Cluster2: E; Cluster3: G. 所以三个
	==> 注意 node 个数有限，而 list 元素数达 million
	==> solution: node 入 set, 对每个set 中元素，找其 next 是否在 set 中，若是 则设 link 为该元素; 若否，则设为自身. 最后为 set 中 link 为自身的元素个数
}

一对 strings k-suspious: 至少有 k-长度的 substring 相同, 给 a set of strings, 找出所有 k-suspious substrings{
==> solution: 用 hash table (Rabin-Karp rolling hash), 其中 string_rolling_hash: val = 0, MUL = 997; for(auto &i:str){ val = (val * MUL + i) % modulers}
--> solution: 用 bloom filter 将所有 substring 分开	
}

找出 5^1234566789893943的从底位开始的1000位数字{
	==> solution: 二进制幂数 + 大数 (1000 位) 乘法
	
}

一个密码锁四位 0-9, 求最短密码串，使得 {// 感想: 数学思维 与 CS 思维的不同
	==> Hamilton 回路 + DFS, 不要想从数学方面找出规则 // 回溯 + 递归 
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

给两个number，m和n，找到m个 >=n 的 最小 Palindromic Numbers{
	==> solution 1: 从 n 开始一个个验证 m*n
	==> solution 2: 生成法 找 n 中间的进位。 分两部分 1. 与 n 同长; 2. 比 n 长的. Sub(int len, deque<int>& result, int num = m+1, int start = n)
		--> 注意 n 本身是 Palindromic, 102 若按 10 变 101 则不行，所以与 n 等长时要对返回的 result 删除部分
}

给一个array of int，以及一个range (low, high), 找出所有 sum 在 range 内的连续的 subsequence{
	==> for L:0 -->n-1; bool IsFromL; 
}
