// ================================================================= //
阅读记录: 2013.10.04, 2013.10.23 2013.11.12
// ================================================================= //
==> 交换数组元素失败，Array[i] 与 Array[Array[i] - 1], 因为过程中下标改变. 如果注意次序即可避免
		* int temp = Array[i]; Array[i] = Array[Array[i]-1]; Array[Array[i]-1] = temp;
		* 用下标不用值：int temp = IndQ[CurInd]; IndQ[CurInd] = Min;	IndQ[Min]    = CurInd; 经常发生在 IndQ 也表示其他下标时		

==> c++ 里 不要用 assert		

==> vector 没有 push, 有 push_back

==> C++ 里用数组初始化 vector: const T a[] = {...}; const vector < const T > V(a,a+sizeof(t)/sizeof(T))
==> 嵌套容器声明但没有赋值时，不能直接使用下标
		e.g. vector< queue<int> > Qs; Qs[i].push(1); 因为 Qs[i] 为空; 记得每次iteration 都清空. 
		
==> queue 没有 clear() 成员

==> inline function 不是 online function		

==> 判断指针是否为 null 时取反了, e.g while(!p){}; 应为 while(p){}

==> 整数类型运算问题： Cnk *= (N-i+1)/i; Cnk *= 1.0 * (N-i+1)/i; Cnk = Cnk *(N-i+1)/i; 三者计算结果皆不同
==> #define INFSmall 1e-8 写成 #define INFSmall 10^-8; 
==> (i >> j & 1) == 1 没有加括号, 运算符优先级

==> 字符串比较时 把 0 当成 '\n'

==> RandomListNode * cur = head, *temp; 定义多个指针时 第二个指针没有加 *

==> 用 map<string,vector<int> >::iterator it = m.find(str) 时，可用 it->second[i] 不可用 (*it)[i]. 当然若知道 str, 则 it[str][i]

==> 直接用数组名而忘记下标 [i]， 一般出现在元素后还接其他成员的情况下

==> multimap 没有运算符 []

==> 循环相关的错误: 1. 变量没有初始化，不能进入循环; 2. 循环结束时，最后一个元素没有进行相关操作; 3. 链表在循环里时 忘记  Link = Link->next;
==> 没有相关操作 map.begin()+1,  map<int,int>::iterator it = m.begin() + 1; 编译出错
==> voltile 告诉编译器每次读取这个变量的时候要直接读原始数据，不读缓存里的。
==> C++ 默认对函数调用自动转换"一次". e.g. GetFoo(Foo i); 而 Foo 有 Foo(int ) 的 constructor. 这里可用 explicit Foo(int) 来限制这种自动转换。 e.g. 若 Foo(int) 与 Foo(char) 同时存在; 则 Foo f = 'c';可会调用 Foo(int), 所以必须加 explicit Foo(int);
==> int Ind[][2] 不能用 int Ind[][], 只有第一维可以省略元素数 
==> string .push_back(char), .append(substr)
==> 等号与 = 混乱, i 与 j 混乱

==> 注意数组越界的问题，如二分法中 (left+right)/2：l + (r-l)<<1;

==> 优秀的程序员都有点懒：他们会舒适地靠背坐着，等待灵感出现，而不急于将自己的第一思想转化成编码。

==> 整数除以2可以用 << 1 代替; 而 (a+b)/2 可以是 a+(b-a)<<1;

==> vector<char> q('c') 出错，应为 vector<char> q(1,'c')

==> 用 iterator 删除元素时，若相邻会影响 e.g. r1.second == r2.first; 若先删除 h.erase(r2.first,r2.second); 再删除 h.erase(r1.first,r1.second); 会出错
==> push_back 写成 puch_back
// ================================================================= //
Gray Code{ // The gray code is a binary numeral system where two successive values differ in only one bit. 2013.08.04 
	==> solution: 运用 递归+镜面原理
    vector<int> grayCode(int n) {
        vector<int> result;
        if(n <= 0){// what if n == 0
            result.push_back(0);
            return result;
        }
        if(n == 1) {// what if n == 1
            result.push_back(0);
            result.push_back(1);
            return result;
        }
        
        result = grayCode(n-1);
        int add = (1<<(n-1));
        for(int n = result.size()-1; n >= 0; --n){
            result.push_back(add+result[n]);
        }
        return result;
    }
}

Decode Ways{ 2013.08.04: A-1, ... Z-26, 给一串数字 计算可以 decode number, 如 12 可以是 AB 也可以是 L, 共两种
	// 错误一: 忘记 '0' 影响
	==> DP: cnt[i] = 0 if str[i] == '0'; else cnt[i] = cnt[i+1]; cnt[i] += cnt[i+2] if i+2 < n && str[i,i+1] fit
	// 解法一: 从开头递归, 大 case 时间不够，因为是 exponential 	
	// 解法二: 从尾开始算，Pre, Cur 是代表所在的间隔位置
	int CountDecodeNum(const char* s, int n){ 
			if(n-- < 1) return 0;
			int Pre = 1;
			int Cur = (s[n] == '0'?0:1);
			while(n-- > 0){
				int New = (s[n] == '0'?0:Cur);
				if(s[n] == '1' || s[n+1] < '7' && s[n] == '2') New += Pre;// 可两位
				Pre = Cur; Cur = New;
				if(Cur == 0 && Pre == 0)break; // 如果两个连续为 0, 则前面的也应全部为零，所以没有必要比较。错误二: 把 Pre 换成 New
			}
			return Cur;
	}
	
	int numDecodings(string s) {
        int size = s.size();
        if(size == 0) return 0;
        int p1, p2, c;
        p1 = 1;
        size -= 1; // the index of last element
        for(int i = size; i >= 0; --i){// from string backward
            if(s[i] == '0') // can not decompose starting with s[i] 
                c = 0;
            else 
                c = p1;
            if(i < size && (s[i] == '1' || s[i] == '2' && s[i+1] < '7')) // can decompose starting with s[i:i+1]
            	 c += p2;
            p2 = p1; p1 = c; // move backward
        }
        return c;
    }
}

Unique Paths{// 统计从 grid 左上--> 到 右下 的 path 数目，其中一些 cell 不能通过
}

Unique Paths II {// 给一个 obstacle matrix, 算从左上角到右下角共有几种走法: 反对角线元素统计
	--> 错误: 
			* 初始化 PathNum 时，没有考虑到 右下角就有 obstacle, 
			* 每次更新 NewPathNum[n-1] 时，没有考虑到旧的 PathNum[n-1] 的 accessible
			
    int uniquePathsWithObstacles(vector<vector<int> > &obstacleGrid) {
        int m = obstacleGrid.size();
        int n = obstacleGrid[0].size();
        
        vector<int> PathNum(n,0);
        PathNum[n-1] = obstacleGrid[m-1][n-1] == 1? 0:1;
        for(int i = n-2; i >= 0; --i)
        {
            PathNum[i] = obstacleGrid[m-1][i] == 1? 0:PathNum[i+1];
        }
        
        for(int i = m-2; i >= 0; --i)
        {
            vector<int> NewPathNum(n,0);
            NewPathNum[n-1] = obstacleGrid[i][n-1] == 1?0:PathNum[n-1];
            for(int j = n-2; j>=0; --j)
            {
                NewPathNum[j] = obstacleGrid[i][j] == 1? 0:NewPathNum[j+1]+PathNum[j];
            }
            PathNum = NewPathNum;
        }
        return PathNum[0];
    }
}

Fabonacci Array{// climb Stairs
  int climbStairs(int n) {// 2013.09.23 无错通过
      int StepNum[2] = {1,2}; // step numbers for n = 1,2
      bool IsFirstNew = n != 1;
      for(int i = 2; i < n; ++i){
          StepNum[IsFirstNew?0:1] = StepNum[0] + StepNum[1];
          IsFirstNew = !IsFirstNew;
      }
      return StepNum[IsFirstNew?1:0];
  }
}

Swap Nodes in Pairs{
  ListNode *swapPairs(ListNode *head) {// 2013.09.23 无错通过
      //if(head == 0 || head->next == 0) return 0;
      ListNode PreHead(0);
      PreHead.next = head;
      head = &PreHead;
      while(head->next && head->next->next){
          ListNode* Second = head->next->next;
          head->next->next = Second->next;
          Second->next = head->next;
          head->next = Second;
          head = head->next->next;
      }
      return PreHead.next;
  }
}

Plus One{// 2013.05.18 注意 reverse, vector 是没有 append(); 2013.09.22 "=" 与 "==" 混乱
    vector<int> plusOne(vector<int> &digits) {
        vector<int> result(digits.rbegin(),digits.rend());
        int c = 1;
        for(int i = 0; i < result.size() && c; ++i){
        	c += result[i];
        	result[i] = c %10;
        	c /- 10;
        }
        if(c) result.push_back(c);
        reverse(result.begin(),result.end());
        return result;
    }
}

// string.c_str() is const char*, can't be char*
string addBinary(string a, string b) {// 出错: 1. 当值有可能是负数时不要用 unsigned. 2. 用 int 型时 i&1 不可判断非零
    string result;
    bool c = 0;
		int ai=a.length()-1;
		int bi=b.length()-1;
    while(ai >= 0 || bi >= 0){// 用 unsigned 导致 ai 很大而越界
        unsigned i,aa,ba;
        aa = a[ai]-'0', ba = b[bi]-'0';
        i = a[ai]-'0' + b[bi]-'0' + c; 
				result.insert(result.begin(),(i==1||i==3)?'1':'0');
        c = (i > 1);
        --ai,--bi;
    }
    if(c)result.insert(result.begin(),'1'); 
    return result;        
}

Spiral Matrix{
Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.
For example,
Given the following matrix:
[
[ 1, 2, 3 ],
[ 4, 5, 6 ],
[ 7, 8, 9 ]
]
You should return [1,2,3,6,9,8,7,4,5].

==> Solution: 
    vector<vector<int> > generateMatrix(int n) {
        vector<vector<int> > result;
        if(n == 0) return result;
        result.assign(n,vector<int> (n,0));
        int Ind[][2] = {{0,1},{1,0},{0,-1},{-1,0}};
        int r = 0, c = -1, k = 1, ii = 0;
        while(n>0){
            int s = n;
            while(s-- > 0){
                r += Ind[ii][0], c += Ind[ii][1];
                result[r][c] = k++;
            }
            if(ii%2 == 0) --n;
            ii = ++ii % 4;
        }
        return result;
   }
    
}

Print the diagnol of a matrix m*n{
给一个矩阵比如
1 2 3 4
5 6 7 8
9 10 11 12

按照反对角线的方式输出
1
2 5
3 6 9
4 7 10
8 11
12

for(int k = 1-m; k < n; ++k){// 三角 k := row - col 
	for(int j = max(i,0)-i; j < m; ++j)printf("%d ",M[i+j][j]);
	printf("\n");
}

for(int i = 0; i < m+n-1; ++i){// 反三角 i := row + col
	for(int j = min(n-1,i); j >= 0;--j)printf("%d ",M[i-j][j]);
	printf("\n");
}

}

打印杨辉三角{ // 右补一，对齐
	1
	2 2
	3 4 3
	4 7 7 4
	
	vector<int> list(n,1);
	for(int i = 0; i < n; ++i){
		for(int j = i; j > 0; --j) list[j] += list[j-1];
		list[0] = i+1;
		for(int j = 0; j <= i; ++j) cout << list[j];
		cout << endl;
	}
}

Simplify Path{ // Given an absolute path for a file (Unix-style), simplify it. 用 queue 来表示

For example,
path = "/home/", => "/home"
path = "/a/./b/../../c/", => "/c"
Corner Cases:

    Did you consider the case where path = "/../"?
    In this case, you should return "/".
    Another corner case is the path might contain multiple slashes '/' together, such as "/home//foo/".
    In this case, you should ignore redundant slashes and return "/home/foo".

==> Solution: use stack, '/' is the determinator of a word, while '..' means pop(), 
int cur = 0, next = 0;
vector<string> stack;
do{
	while(path[next] != '\0' && path[next] != '/') ++next;
	if(next == cur)break;
	string s(&path+cur, &path+next);
	if(s == ".."){
		if(!stack.empty()){
			stack.pop_back();
		}else{
			err << "Illegal path: " << p << endl;
			return;
		}
	}else{
		stack.push_back(s);
	}	
	cur == next;
}while(path[cur] != '\0');

}

reverseInt(int n){ // reverse an integer,  be care of of out range, use double type and deque 
	bool IsNegative = n < 0;
	n = IsNegative?-n:n; // 最好 used unsigned long long, 不然会有溢出错误
	int i = 0;
	while(n>0){// 2013.09.26 improve, does not need to use deque
		int t = n%10;
		i = i * 10 + t;
		if(i < 0){// out of range
			cerr<< "Flipped interger is to large!\n";
			exit(1);
		}
		n /= 10;
	}	
	return IsNegative?-i:i;
}

int divide(int dend, int dor) {// 不用 * mod / 实现除法, bit operations
	if(dend == 0 || dor == 0) return 0;
	
	bool IsNegative = (dend < 0) && (dor > 0) || (dend > 0) && (dor < 0);
	dend = dend > 0 ? dend: -dend;
	dor  = dor > 0 ? dor: -dor;
	
	// align == multiply 2
	int k;
	for(k = 0; dend > dor; ++k) dor <<= 1; // 注意数组越界
	
	// calculate
	int result = 0;
	do{
		if(dend>=dor){
			dend -= dor;
			result &= (1 << k);
		}
		dor >>= 1;
		--k;	
	}while(k >= 0);
	
	return IsNegative?result:-result;
}

Subsets{// 可由此引申 enumerate all possible combinations 其实与 enumeration 相关，2 进制
    vector<vector<int> > subsets(vector<int> &S) {
        vector<vector<int> > result;
        sort(S.begin(),S.end());
        
        vector<bool> isInSet(S.size(),false);
        int CurInd = S.size();
        while(CurInd >= 0){
        	vector<int> V;
            if(CurInd == S.size() -1 ){
                V.clear();
                for(int i = 0; i < S.size(); ++i){
                    if(isInSet[i])V.push_back(S[i]); // 出错：忘记isInSet 的意义，用isInSet[i] 代替 isInSet[i+1] 
                }
                result.push_back(V);                
            }            
            // flip carry-on                
            if(isInSet[CurInd] == false){
                isInSet[CurInd] = true;
                CurInd = S.size();
            }else{
                isInSet[CurInd] = false;
                CurInd -= 1;
            }
        }
        return result;
   }
}

Subsets II{// 有重复元素的，要求每个 subset 里元素非降序, 每位进制为该元素出现次数
	==> 错误: 要动用到 2008 Visual 来查错!
			* vector<int> UniqSet(S.begin(),1); 不可用 S.begin(), 应用 S[0];
			* SameElementNum/UniqSet 的index 与 Index 下标错位一，所以 Index[i] 对应的应该是 UniqSet[i-1];
			* 不用 Index 与 Sam3eElementNum/UniqSet 的错位，通过下标判断，这样就可以减少下标越界的问题
			* UniqSet(1,S[0]); 两个参数的位置互换 UniqSet(S[0],1); 这种错误出现在参数类型相同且意义相近的情况下
			
    vector<vector<int> > subsetsWithDup(vector<int> &S) {
        vector<vector<int> > result;
        if(S.empty()) return result;
        sort(S.begin(),S.end());
        
        vector<int> SameElementNum(1,1);
        vector<int> UniqSet(1,S[0]);
        int ElementArraySize = 1;
        for(int i = 1; i <S.size(); ++i){
            if(S[i] != S[i-1]){
                SameElementNum.push_back(1);
                UniqSet.push_back(S[i]);
                ElementArraySize += 1;
            }else{
                SameElementNum[ElementArraySize-1] += 1;
            }
        }
        
        vector<int> Index(ElementArraySize,0);
        int CurInd = ElementArraySize-1;  
        while(CurInd >= 0){
            if(CurInd == ElementArraySize-1){
                vector<int> V;
                V.clear();
                for(int i = 0; i < ElementArraySize; ++i){
                    int temp = Index[i];
                    while(--temp>=0) V.push_back(UniqSet[i]);                         
                }
                result.push_back(V);                
            }            
            // flip carry-on                
            if(Index[CurInd] < SameElementNum[CurInd]){
                Index[CurInd] += 1;
                CurInd = ElementArraySize-1;                
            }else{
                Index[CurInd] = 0;
                CurInd -= 1;
            }
        }
        return result;
   }	
}

nChooseK{// 2013.05.21 for research ENU, 没有重复元素。虽然不是 permuation, 但也是进制问题: Sample[i] 为所选元素里第 i 大的 
	void nChooseK(int n,int k){// index(number) from 0 to n-1, 访问次序是 lexicographic minimum, 
		if(k <= 0 || k > n) return;
		int*Sample = new int[k];
		int i;
		for(i = 0; i < k; ++i) Sample[i] = i;// starting from 0,...,k-1
		i = k - 1;
		while(i >= 0){
			if(i == k - 1){
				for(int s = 0; s < k; ++s) cout<<Sample[s] << " "; cout << endl;
			}
			Sample[i] += 1; // move 1
			if(Sample[i] > i+(n-k)){// because at most left k-i element, we have Sample[i] <= (n-k)+i
				i -= 1;// carry on to previous bit. If i == 0, it will break the loop
			}else{// do not need to carry on
				while(i < k -1){
					Sample[i+1] = Sample[i] + 1;
					++i;
				}
			}
		}
		delete[] Sample;
	}
}


Permutation{
	random permutation{
		for(i = n-1; i > 0; ++i){
				d = rand{0,...,n-1};
				swap a[d] and a[i-1]
		}
	}
	
  void permute_sub(vector<int> &num,int cur, vector<vector<int> > &result){
      if(cur == num.size()-1){
          result.push_back(num);
          return;
      }
      permute_sub(num,cur+1, result);
      for(int i = 1; i < num.size()-cur;++i){
          if(num[cur] != num[cur+i]){
              int j = num[cur+i]; num[cur+i] = num[cur]; num[cur] = j;
              permute_sub(num,cur+1, result);
              j = num[cur+i]; num[cur+i] = num[cur]; num[cur] = j;
          }
      }
  }
  vector<vector<int> > permute(vector<int> &num) {// recursive
      // Note: The Solution object is instantiated only once and is reused by each test case.
      vector< vector<int> > result;
      int s = num.size();
      if(s == 0) return result;
      permute_sub(num,0, result);
      return result;
  }

Generate all permutation (next is lexicographically minimal permutation){
==> 1. sort a[...] in increasing order.
		2. Find the largest index k such that a[k] < a[k + 1] (若有重复元素，用 <= instead of < ). If no such index exists, the permutation is the last permutation.
    3. Find the largest index l such that a[k] <= a[l] (若有重复元素，用 <= instead of < ). Since k + 1 is such an index, l is well defined and satisfies k < l.
    4. Swap a[k] with a[l].
    5. Reverse the sequence from a[k + 1] up to and including the final element a[n].
==> 几个注意的地方: 1. 若 s == 最后一个元素如何处理. 2. 重复元素如何办：用等号判断 num[k] >= num[k+1], l 从 k+1 到末尾
    vector<vector<int> > permute(vector<int> &num) {
        vector< vector<int> > result;
        int s = num.size();
        if(s == 0) return result;
        sort(num.begin(),num.end());
        do{
            result.push_back(num);
            int k,l,i;
            for(k = s-2; k >= 0 && num[k] >= num[k+1]; --k);// find the largest k, s.t. A[k] < A[k+1]
            if(k < 0) return result;
            for(l = k+1; l < n && num[l] > num[k]; --l);
            l -= 1;
            i = num[k]; num[k] = num[l]; num[l] = i;
            for(l = s-1, k += 1; l > k; --l, ++k){
               i = num[k]; num[k] = num[l]; num[l] = i; 
            }
        }while(1);
    }
    
After step 1, one knows that all of the elements strictly after position k form a weakly decreasing sequence, so no permutation of these elements will make it advance in lexicographic order; to advance one must increase a[k]. Step 2 finds the smallest value a[l] to replace a[k] by, and swapping them in step 3 leaves the sequence after position k in weakly decreasing order. Reversing this sequence in step 4 then produces its lexicographically minimal permutation, and the lexicographic successor of the initial state for the whole sequence.
==> Extension: find the "Previous"  lexicographically minimal permutation
		1. find largest k s.t. A[k] > A[k + 1], find 
}

Generate all permutation (next is swapping two adjacent values) Steinhaus–Johnson–Trotter algorithm{
	==> 递归的思想：用最大元素作为间隔，共 n * f(n-1) 次，正好 n! 需要标记最大元素位置, 而其辅助数组正好此功能
==> 1. sort a[...] in increasing order, Sign[0] = 0; Sign[1...n-1] = negative
		2. 找最大的据有 a nonzero direction 的数，and swaps it in the indicated direction (one position)
				注意 1 -2 -3 -3 --> "1 -3 -2 -3 --> 1 -3 -3 -2" --> ... --> +3 +3 2 1 --> "+3 2 +3 1 --> 2 +3 +3 1" 重复元素移动不成功
		3. If this causes the chosen element to reach the first or last position within the permutation, or if the next element in the same direction is larger than the chosen element, the direction of the chosen element is set to zero.
				e.g. 1 -4 -2 -3 --> 4 1 -2 -3; 1 -3 +4 -2 --> 1 -3 -2 4; 3 1 -2 --> +3 2 1, because 3 > 2;
		4. After each step, all elements greater than the chosen element have their directions set to positive or negative, according to whether they are concentrated at the start or the end of the permutation respectively. 
			e.g. 4 3 1 -2 --> +3 +4 2 1; 
   vector<vector<int> > permute(vector<int> &num) {// smallest distance
        vector< vector<int> > result;
        int i,s = num.size();
        if(s == 0) return result;
        sort(num.begin(),num.end());
        vector<int> sign(s,-1); 
				for(sign[0] = 0, i = 1; i < s && num[i] == num[i-1]; ++i) sign[i] = 0;
				vector<int> &test = sign;
        do{
            result.push_back(num);
            int k,m;
            for(k = 0; k < s && sign[k] == 0; ++k);// 找最大 index k 时出错
            if(k == s) return result;
            for(i = k+1; i < s ; ++i) if(num[i] > num [k] && sign[i] != 0) k = i;
            m = k + sign[k];
            i = num[m]; num[m] = num[k]; num[k] = i;
            i = sign[m]; sign[m] = sign[k]; sign[k] = i;
            if((sign[m] == -1) && (m < 1 || num[m-1] >= num[m]) || (sign[m] == 1) && (m > s-2 || num[m+1] >= num[m])) sign[m] = 0;
            for(k = 0; k < s; ++k) if(num[k] > num[m]) sign[k] = (k<m)-(k>m);
        }while(1);
    }
}
}

对一 array A[n] 按 给定的index permutation P[i] 生成新 array, 要求 in place{
	for(int i = 0; i < n ; ++i){
		if(P[i] >= 0){
			int NInd = p[i], NVal = A[i];
			P[i] = -P[i]; // 活动过的归为 -1. 
			do{
				int t = A[NInd];
				A[NInd] = NVal;
				NVal = t;
				NInd = P[NInd];
			}while(P[NInd] != 1);
		}
	}
	==> Extension: 给一 permutation P[n], 求其逆 permutation: 因为 P[P[i]] = i; 令 A[i] == i 即可
}


count the permutation index of a given sample{
long PermutationIndex(int * Array, int AlphabetNum, int * Sample){
	// find the index of a given sample in increasing order, allowing duplicate elements
	// Alphabet[Sample[i]] is the ith character of the given sample
	// Array[i] is the number of Alphabet[i] appared in that sample, Array[i] can be 0;
	// length of Sample = sum of Array

	int SampleLen = 0;
	int CurrPos = AlphabetNum;
	while(--CurrPos >= 0)SampleLen += Array[CurrPos];
	
	long Index = 0;
	for(CurrPos = 0; CurrPos < SampleLen; ++CurrPos){
		for(int i = 0; i < Sample[CurrPos]; ++i){
			if(Array[i] > 0){
				Array[i] -= 1;
				Index += PermutationCount(Array, AlphabetNum);
				Array[i] += 1;
			}
		}
		Array[Sample[CurrPos]] -= 1;
	}
	
	return Index;
}

long PermutationCount(int* Array, int AlphabetNum){
	// count the number of permutation given an array.
	long Count = 1;
	long TotalNum = 0;
	while(--AlphabetNum>=0)
		if(Array[AlphabetNum] > 0)
			for(int i = 1; i <= Array[AlphabetNum]; ++i)
				Count *= ++TotalNum / i;
	return Count;
}
}

Place n-queens in a n*n grid{// 
	==> define path as path[i] is the col index of row i: 变成 permutation 问题，只不过可以提前知道可行性，不用check 所有可能的 permutation
	==> 上法想复杂了，其实用的是 recursive，根据 path.size() 及 path 内容，每行找出可行的 element 每个都试一下
	==> 另一错误想法是 用一个 vector<bool> available，如果没有行号，不能update; 但要用到复杂的 bit (三个变量 列,对角，反对角)操作来实现 
	
}

一个数字可以对应0-n个字母,给一串数字，列出所有可能的字母组合(0表示不用输出){
==> enumeration: 可用 recursive 来做。但也可用 iterate 来做。思路: 将所有可能字母排序，用 Size[StrLen] 表示进制，Index[StrLen] 表示现组合对应字母下标 

vector< vector<char> > string;
vector<int> Pos(string.size(),0);
PrintAll(string,Pos,0,string.size());
void Print_Recursive(vector< vector<char> > & string, vector<int> & Pos, int Cur){// recursive version
		if(Cur == string.size()){
			cout << string(Pos);
			return;
		}
		for(int i = 0; i < string[Cur].size(); ++i){
				Pos[Cur] = i;
				PrintAll(string,Pos,Cur+1,string.size());
		}		
}
void Print_Iterative(vector< vector<char> > & string){
	int len = string.size();
	vector<int> Pos(len,0);
	int Cur = len;
	while(Cur >= 0){
		if(Cur == len){// Cur == len is a token of end
			cout << string(Pos);
			Cur -= 1;
			Pos[Cur] += 1;
		}
				
		if(Pos[Cur] == string.size[Cur].size()){
			Pos[Cur] = 0;
			if(Cur == 0) break;
			Pos[--Cur] += 1;
		}else{
			Cur += 1;
		}
	}
}
}

ZigZag Conversion{
The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)

P   A   H   N
A P L S I I G
Y   I   R

And then read line by line: "PAHNAPLSIIGYIR"

==> 
string convert(string s, int nRows) {
	string result = s;
	if(nRows <= 1) return s;
	int OddColInc = nRows + nRows/2;
	int TotalSize = s.size();
	int r = 0;
	int ri = 0;
	while(r < nRows){
		bool IsEvenRow = r & 1;// start from zero
		int si = r;
		while(si < TotalSize){
			result[ri++] = s[si];
			if(IsEvenRow){
				int i = si+nRows+(r-1)/2;
				if(i < TotalSize)result[ri++] = s[i];
			}
			si += OddColInc;
		}
		r += 1;
	}
}

    string convert(string s, int nRows) {// s.length() >= nRows
        string result;
        for(int r = 0, step = nRows+nRows/2, halfstep = nRows-1; r < nRows; ++r){
            int i = r;
            do{
                result.push_back(s[i]);
                if(r%2 == 1 && i + halfstep < s.length()) result.push_back(s[i+halfstep]);
                i += step;
            }while(i < s.length());
        }
        return result;
    }
}

Valid Parentheses{// 
	==> solution: stack
    bool isValid(string s) {
        vector<char> status;
        for(int i = 0; i < s.size(); ++i){
            switch(s[i]){
                case '(': case '{': case '[': status.push_back(s[i]); break;
                case ')': if(status.size() == 0 || status.back() != '(') return false; status.pop_back(); break;
                case '}': if(status.size() == 0 || status.back() != '{') return false; status.pop_back(); break;
                case ']': if(status.size() == 0 || status.back() != '[') return false; status.pop_back(); break;
            }
        }
        return status.size() == 0; 
    }
}


Determine if a Sudoku is valid 2012.10.23{
==> first bug free in first trial.
    bool isValidSudoku(vector<vector<char> > &board) {
        const int n = 9;
        for(int i = 0; i < n; ++i)
            for(int j = 0; j < n; ++j){
                if(board[i][j] == '.') continue;
                for(int k = 0; k < n; ++k){
                    if(k == j || board[i][k] == '.') continue;
                    if(board[i][k] == board[i][j]) return false;
                }
                for(int k = 0; k < n; ++k){
                    if(k == i || board[k][j] == '.') continue;
                    if(board[k][j] == board[i][j]) return false;
                }
                int x = i / 3;  x = x * 3 + 3;
                int y = j / 3;  y = y * 3 + 3;
                
                for(int k = x-3; k < x; ++k)
                    for(int l = y-3; l < y; ++l){
                        if(k == i && l == j || board[k][l] == '.') continue;
                        if(board[k][l] == board[i][j]) return false;
                    }
            }
        return true;    
    }
}

solve a Sudoku{
Write a program to solve a Sudoku puzzle by filling the empty cells. 2012.10.23
==> solution: 用回溯法, recursive function; 
	* 定义一个新结构 Pos{int r,c,cnt; bool p[9];} 及 Pos* p[9][9], 若定下来则为 0, 及 deque<Pos> q, 
	* 找一个有最小可能的末定元素，用 make_heat(q.begin(),q.end()) 排序， 若只有一种可能，则安排，将其 pop_front(), 若遇到其他 cnt == 0 的，表明不可靠。若有多种可能 call function. 
	* 最好用函数的值传递，不用 new. 由于 Pos* p 是指针，在 function 的开关要与 实参 对应起来(这是为什么 Pos 需要 int r,c 的原因)
	
}

to generate sudoku{
Write a program to generate sudoku. 2012.10.25

==> Solution:
(1) Basic block B 3*3 and transition matrix T, can be three generator [B1 B2 B3; B1T B2T B3T; B1TT B2TT B3TT];
(2) Permution Matrix T 3*3 (can be two possible since 1:3 has 3 permutations, except the one no change). Property T2 = T*T; T*T2 = I; T =T2;
(3) [B TB TTB; BT TBT TTBT; BTT TBTT TTBTT]; done
(4) Other operation: 3*3 blocks permution (row and col)
(5) rows swap (in the same row of blocks)
(6) col swap

==> Another solution(More flexible):
(1) Generate diagonal blocks (3)
(2) Calculate RestNum, 
(3) start from the element with the smallest RestNum, filled with any of possible number
(4) update RestNum, repeat (3) until all blanks are filled
}

Determine If Two Rectangles Overlap given 对角点 (P1,P2) (P3,P4){
	==> solution: 想想不交的条件是什么，再取反即可
}

Given an integer N, print numbers from 1 to N in lexicographic order{
	==> solution: 使用递归，p 为前缀。
	void Print(int p, int n){
		if(p > n) return; // 防止打印出的 n == 0 的情况
		cout << p << endl;
		for(int i = 0; i <= 9; ++i){// p*10 + i 为新的前缀
			if(p*10 + i <= n) Print(p*10+i,n);
		}
		if(p < 9) Print(p+1,n); // 当 p 为个位时上升
	}  
}

打印出小于 n 的所有 prime {
	void PrintPrime(int n){
		vector<bool> b(n+1,true);
		b[0] =b[1] = false;
		int c = 2;
		while(c <= n){
			cout << c << endl;
			int t = c<<1;
			while(t <= n){ b[t] = false; t+=c;}
			do{
			++c;
			}while(c <= n && b[c] == false);
		}
	}
}


getNthPrime{// 找第 n 个素数 
	==> solution: 维护一个 两个 queues： Prime, MaxNumPrime 这样遇到一个新数，不用通过 divide 来判断是否素数，只要在 MaxNumofPrime 里加减即可
	int getNthPrime(int n){// n should >= 1
		vector<int> Prime(1,2), MaxNumPrime(1,2);
		int CurInt = 3;
		while(Prime.size() < n){
			bool IsPrime = true;
			for(int i = 0; i < Prime.size(); ++i){
				if(CurInt == MaxNumPrime[i] + Prime[i]){
						MaxNumPrime[i] = CurInt;
						IsPrime = false;
				}
			}
				if(IsPrime){
					Prime.push_back(CurInt);
					MaxNumPrime.push_back(CurInt);
				}
				++CurInt;
			}
		return Prime.back();
	}
}