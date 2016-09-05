// ================================================================= //
�Ķ���¼: 2013.10.04, 2013.10.23 2013.11.12
// ================================================================= //
==> ��������Ԫ��ʧ�ܣ�Array[i] �� Array[Array[i] - 1], ��Ϊ�������±�ı�. ���ע����򼴿ɱ���
		* int temp = Array[i]; Array[i] = Array[Array[i]-1]; Array[Array[i]-1] = temp;
		* ���±겻��ֵ��int temp = IndQ[CurInd]; IndQ[CurInd] = Min;	IndQ[Min]    = CurInd; ���������� IndQ Ҳ��ʾ�����±�ʱ		

==> c++ �� ��Ҫ�� assert		

==> vector û�� push, �� push_back

==> C++ ���������ʼ�� vector: const T a[] = {...}; const vector < const T > V(a,a+sizeof(t)/sizeof(T))
==> Ƕ������������û�и�ֵʱ������ֱ��ʹ���±�
		e.g. vector< queue<int> > Qs; Qs[i].push(1); ��Ϊ Qs[i] Ϊ��; �ǵ�ÿ��iteration �����. 
		
==> queue û�� clear() ��Ա

==> inline function ���� online function		

==> �ж�ָ���Ƿ�Ϊ null ʱȡ����, e.g while(!p){}; ӦΪ while(p){}

==> ���������������⣺ Cnk *= (N-i+1)/i; Cnk *= 1.0 * (N-i+1)/i; Cnk = Cnk *(N-i+1)/i; ���߼������Բ�ͬ
==> #define INFSmall 1e-8 д�� #define INFSmall 10^-8; 
==> (i >> j & 1) == 1 û�м�����, ��������ȼ�

==> �ַ����Ƚ�ʱ �� 0 ���� '\n'

==> RandomListNode * cur = head, *temp; ������ָ��ʱ �ڶ���ָ��û�м� *

==> �� map<string,vector<int> >::iterator it = m.find(str) ʱ������ it->second[i] ������ (*it)[i]. ��Ȼ��֪�� str, �� it[str][i]

==> ֱ�����������������±� [i]�� һ�������Ԫ�غ󻹽�������Ա�������

==> multimap û������� []

==> ѭ����صĴ���: 1. ����û�г�ʼ�������ܽ���ѭ��; 2. ѭ������ʱ�����һ��Ԫ��û�н�����ز���; 3. ������ѭ����ʱ ����  Link = Link->next;
==> û����ز��� map.begin()+1,  map<int,int>::iterator it = m.begin() + 1; �������
==> voltile ���߱�����ÿ�ζ�ȡ���������ʱ��Ҫֱ�Ӷ�ԭʼ���ݣ�����������ġ�
==> C++ Ĭ�϶Ժ��������Զ�ת��"һ��". e.g. GetFoo(Foo i); �� Foo �� Foo(int ) �� constructor. ������� explicit Foo(int) �����������Զ�ת���� e.g. �� Foo(int) �� Foo(char) ͬʱ����; �� Foo f = 'c';�ɻ���� Foo(int), ���Ա���� explicit Foo(int);
==> int Ind[][2] ������ int Ind[][], ֻ�е�һά����ʡ��Ԫ���� 
==> string .push_back(char), .append(substr)
==> �Ⱥ��� = ����, i �� j ����

==> ע������Խ������⣬����ַ��� (left+right)/2��l + (r-l)<<1;

==> ����ĳ���Ա���е��������ǻ����ʵؿ������ţ��ȴ���г��֣��������ڽ��Լ��ĵ�һ˼��ת���ɱ��롣

==> ��������2������ << 1 ����; �� (a+b)/2 ������ a+(b-a)<<1;

==> vector<char> q('c') ����ӦΪ vector<char> q(1,'c')

==> �� iterator ɾ��Ԫ��ʱ�������ڻ�Ӱ�� e.g. r1.second == r2.first; ����ɾ�� h.erase(r2.first,r2.second); ��ɾ�� h.erase(r1.first,r1.second); �����
==> push_back д�� puch_back
// ================================================================= //
Gray Code{ // The gray code is a binary numeral system where two successive values differ in only one bit. 2013.08.04 
	==> solution: ���� �ݹ�+����ԭ��
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

Decode Ways{ 2013.08.04: A-1, ... Z-26, ��һ������ ������� decode number, �� 12 ������ AB Ҳ������ L, ������
	// ����һ: ���� '0' Ӱ��
	==> DP: cnt[i] = 0 if str[i] == '0'; else cnt[i] = cnt[i+1]; cnt[i] += cnt[i+2] if i+2 < n && str[i,i+1] fit
	// �ⷨһ: �ӿ�ͷ�ݹ�, �� case ʱ�䲻������Ϊ�� exponential 	
	// �ⷨ��: ��β��ʼ�㣬Pre, Cur �Ǵ������ڵļ��λ��
	int CountDecodeNum(const char* s, int n){ 
			if(n-- < 1) return 0;
			int Pre = 1;
			int Cur = (s[n] == '0'?0:1);
			while(n-- > 0){
				int New = (s[n] == '0'?0:Cur);
				if(s[n] == '1' || s[n+1] < '7' && s[n] == '2') New += Pre;// ����λ
				Pre = Cur; Cur = New;
				if(Cur == 0 && Pre == 0)break; // �����������Ϊ 0, ��ǰ���ҲӦȫ��Ϊ�㣬����û�б�Ҫ�Ƚϡ������: �� Pre ���� New
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

Unique Paths{// ͳ�ƴ� grid ����--> �� ���� �� path ��Ŀ������һЩ cell ����ͨ��
}

Unique Paths II {// ��һ�� obstacle matrix, ������Ͻǵ����½ǹ��м����߷�: ���Խ���Ԫ��ͳ��
	--> ����: 
			* ��ʼ�� PathNum ʱ��û�п��ǵ� ���½Ǿ��� obstacle, 
			* ÿ�θ��� NewPathNum[n-1] ʱ��û�п��ǵ��ɵ� PathNum[n-1] �� accessible
			
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
  int climbStairs(int n) {// 2013.09.23 �޴�ͨ��
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
  ListNode *swapPairs(ListNode *head) {// 2013.09.23 �޴�ͨ��
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

Plus One{// 2013.05.18 ע�� reverse, vector ��û�� append(); 2013.09.22 "=" �� "==" ����
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
string addBinary(string a, string b) {// ����: 1. ��ֵ�п����Ǹ���ʱ��Ҫ�� unsigned. 2. �� int ��ʱ i&1 �����жϷ���
    string result;
    bool c = 0;
		int ai=a.length()-1;
		int bi=b.length()-1;
    while(ai >= 0 || bi >= 0){// �� unsigned ���� ai �ܴ��Խ��
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
��һ���������
1 2 3 4
5 6 7 8
9 10 11 12

���շ��Խ��ߵķ�ʽ���
1
2 5
3 6 9
4 7 10
8 11
12

for(int k = 1-m; k < n; ++k){// ���� k := row - col 
	for(int j = max(i,0)-i; j < m; ++j)printf("%d ",M[i+j][j]);
	printf("\n");
}

for(int i = 0; i < m+n-1; ++i){// ������ i := row + col
	for(int j = min(n-1,i); j >= 0;--j)printf("%d ",M[i-j][j]);
	printf("\n");
}

}

��ӡ�������{ // �Ҳ�һ������
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

Simplify Path{ // Given an absolute path for a file (Unix-style), simplify it. �� queue ����ʾ

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
	n = IsNegative?-n:n; // ��� used unsigned long long, ��Ȼ�����������
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

int divide(int dend, int dor) {// ���� * mod / ʵ�ֳ���, bit operations
	if(dend == 0 || dor == 0) return 0;
	
	bool IsNegative = (dend < 0) && (dor > 0) || (dend > 0) && (dor < 0);
	dend = dend > 0 ? dend: -dend;
	dor  = dor > 0 ? dor: -dor;
	
	// align == multiply 2
	int k;
	for(k = 0; dend > dor; ++k) dor <<= 1; // ע������Խ��
	
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

Subsets{// ���ɴ����� enumerate all possible combinations ��ʵ�� enumeration ��أ�2 ����
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
                    if(isInSet[i])V.push_back(S[i]); // ��������isInSet �����壬��isInSet[i] ���� isInSet[i+1] 
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

Subsets II{// ���ظ�Ԫ�صģ�Ҫ��ÿ�� subset ��Ԫ�طǽ���, ÿλ����Ϊ��Ԫ�س��ִ���
	==> ����: Ҫ���õ� 2008 Visual �����!
			* vector<int> UniqSet(S.begin(),1); ������ S.begin(), Ӧ�� S[0];
			* SameElementNum/UniqSet ��index �� Index �±��λһ������ Index[i] ��Ӧ��Ӧ���� UniqSet[i-1];
			* ���� Index �� Sam3eElementNum/UniqSet �Ĵ�λ��ͨ���±��жϣ������Ϳ��Լ����±�Խ�������
			* UniqSet(1,S[0]); ����������λ�û��� UniqSet(S[0],1); ���ִ�������ڲ���������ͬ����������������
			
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

nChooseK{// 2013.05.21 for research ENU, û���ظ�Ԫ�ء���Ȼ���� permuation, ��Ҳ�ǽ�������: Sample[i] Ϊ��ѡԪ����� i ��� 
	void nChooseK(int n,int k){// index(number) from 0 to n-1, ���ʴ����� lexicographic minimum, 
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
		2. Find the largest index k such that a[k] < a[k + 1] (�����ظ�Ԫ�أ��� <= instead of < ). If no such index exists, the permutation is the last permutation.
    3. Find the largest index l such that a[k] <= a[l] (�����ظ�Ԫ�أ��� <= instead of < ). Since k + 1 is such an index, l is well defined and satisfies k < l.
    4. Swap a[k] with a[l].
    5. Reverse the sequence from a[k + 1] up to and including the final element a[n].
==> ����ע��ĵط�: 1. �� s == ���һ��Ԫ����δ���. 2. �ظ�Ԫ����ΰ죺�õȺ��ж� num[k] >= num[k+1], l �� k+1 ��ĩβ
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

Generate all permutation (next is swapping two adjacent values) Steinhaus�CJohnson�CTrotter algorithm{
	==> �ݹ��˼�룺�����Ԫ����Ϊ������� n * f(n-1) �Σ����� n! ��Ҫ������Ԫ��λ��, ���丨���������ô˹���
==> 1. sort a[...] in increasing order, Sign[0] = 0; Sign[1...n-1] = negative
		2. �����ľ��� a nonzero direction ������and swaps it in the indicated direction (one position)
				ע�� 1 -2 -3 -3 --> "1 -3 -2 -3 --> 1 -3 -3 -2" --> ... --> +3 +3 2 1 --> "+3 2 +3 1 --> 2 +3 +3 1" �ظ�Ԫ���ƶ����ɹ�
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
            for(k = 0; k < s && sign[k] == 0; ++k);// ����� index k ʱ����
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

��һ array A[n] �� ������index permutation P[i] ������ array, Ҫ�� in place{
	for(int i = 0; i < n ; ++i){
		if(P[i] >= 0){
			int NInd = p[i], NVal = A[i];
			P[i] = -P[i]; // ����Ĺ�Ϊ -1. 
			do{
				int t = A[NInd];
				A[NInd] = NVal;
				NVal = t;
				NInd = P[NInd];
			}while(P[NInd] != 1);
		}
	}
	==> Extension: ��һ permutation P[n], ������ permutation: ��Ϊ P[P[i]] = i; �� A[i] == i ����
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
	==> define path as path[i] is the col index of row i: ��� permutation ���⣬ֻ����������ǰ֪�������ԣ�����check ���п��ܵ� permutation
	==> �Ϸ��븴���ˣ���ʵ�õ��� recursive������ path.size() �� path ���ݣ�ÿ���ҳ����е� element ÿ������һ��
	==> ��һ�����뷨�� ��һ�� vector<bool> available�����û���кţ�����update; ��Ҫ�õ����ӵ� bit (�������� ��,�Խǣ����Խ�)������ʵ�� 
	
}

һ�����ֿ��Զ�Ӧ0-n����ĸ,��һ�����֣��г����п��ܵ���ĸ���(0��ʾ�������){
==> enumeration: ���� recursive ��������Ҳ���� iterate ������˼·: �����п�����ĸ������ Size[StrLen] ��ʾ���ƣ�Index[StrLen] ��ʾ����϶�Ӧ��ĸ�±� 

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
==> solution: �û��ݷ�, recursive function; 
	* ����һ���½ṹ Pos{int r,c,cnt; bool p[9];} �� Pos* p[9][9], ����������Ϊ 0, �� deque<Pos> q, 
	* ��һ������С���ܵ�ĩ��Ԫ�أ��� make_heat(q.begin(),q.end()) ���� ��ֻ��һ�ֿ��ܣ����ţ����� pop_front(), ���������� cnt == 0 �ģ��������ɿ������ж��ֿ��� call function. 
	* ����ú�����ֵ���ݣ����� new. ���� Pos* p ��ָ�룬�� function �Ŀ���Ҫ�� ʵ�� ��Ӧ����(����Ϊʲô Pos ��Ҫ int r,c ��ԭ��)
	
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

Determine If Two Rectangles Overlap given �Խǵ� (P1,P2) (P3,P4){
	==> solution: ���벻����������ʲô����ȡ������
}

Given an integer N, print numbers from 1 to N in lexicographic order{
	==> solution: ʹ�õݹ飬p Ϊǰ׺��
	void Print(int p, int n){
		if(p > n) return; // ��ֹ��ӡ���� n == 0 �����
		cout << p << endl;
		for(int i = 0; i <= 9; ++i){// p*10 + i Ϊ�µ�ǰ׺
			if(p*10 + i <= n) Print(p*10+i,n);
		}
		if(p < 9) Print(p+1,n); // �� p Ϊ��λʱ����
	}  
}

��ӡ��С�� n ������ prime {
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


getNthPrime{// �ҵ� n ������ 
	==> solution: ά��һ�� ���� queues�� Prime, MaxNumPrime ��������һ������������ͨ�� divide ���ж��Ƿ�������ֻҪ�� MaxNumofPrime ��Ӽ�����
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