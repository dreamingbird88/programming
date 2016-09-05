// ================================================================= //
�Ķ���¼: 2013.10.04 2013.10.24
// ================================================================= //

�ı��β���ʽ --> DP ����

һ���������飬����Ϊn�������Ϊm�ݣ�ʹ���ݵĺ���ȣ���m�����ֵ{// �� DP ???
  ����{3��2��4��3��6} ���Էֳ�{3��2��4��3��6} m=1;
  {3,6}{2,4,3} m=2
  {3,3}{2,4}{6} m=3 ����m�����ֵΪ3
  
	==> solution:
		1. �� Max �� Sum ���� sum ��ʽ�ֽ⡣���� Max ӦΪ���ӵ��½�, �� m <= Sum/Max; �ֽ� F[i] with A[i] �η���, i = 0,...,f-1
		2. �� merge f-sort array �� queue ��������С����õ��ֽ����ӣ�ÿ�����Ӷ�Ӧ�ֽ��ÿ��֮�ͣ����� < Max ����
		3. �������� O(nlgn), ��ÿ�����ӣ�������
}

Given an array���ֳ����ݣ�ʹ����֮�;������{// �������� 
	==> solution 0: Backpack problem f_i(x) = min{f_{i-1}(x), f_{i-1}(x-a[i]), |x-a[i]|} where x is sum/2; ʵ��������
	==> solution 2: �Ƚ��� ����Ϊ sum / 2 �� bool �� (���� set ��������)���Ӵ�С����Ԫ��(�������)����ÿ�� a[i], ��� set �е� sum s��û�� ���� s + a[i] == sum/2, ������ return true
	==> Extension: ʹ������������� f_i(x,n) = min{f_{i-1}(x,n), f_{i-1}(x-a[i],n-1), |x-a[i]| if n == 1} with n is num/2;
}

�������� x ����Ҹ�����ֵ {k1, k2, ..., km}����{// �������� 
	==> ���Ż��ҽ�Ϸ�����ʹ���������Ĳ�ֵ��С e.g. $417 50c/10c/7c 
	--> Solution: p_m[i] = min{min{p_{m-1}[i-k_m * s]}, V%k_m}, i = 0,...,x the least overpaid value
	==> �����ٻ������ sum: ��sum �Ի��Ҷ����ƻ������� 01 ���� f_{i}(s) {k1, k2, ..., ki} ��� sum == s ����С�� ע��Ӧ����С������ʼ
}

Knackpack01{
	bool Knackpack01(int space[], int value[], int Num, int TotalSpace){
		vector<bool> IsPacked(TotalSpace+1,false); 
		IsPacked[0] = true;
		int Total = 0; 
		while(Num-- >0){
			Total = min(Total+space[Num],TotalSpace);
			for(int i = Total; i >= space[Num]; --i)IsPacked[i] = IsPacked[i-space[Num]];// can not from space[Num] to Total since douple counted
		}
		return IsPacked[TotalSpace];
	}
}

Josephus problem n �����֣�0,1,��,n-1���γ�һ��ԲȦ��������0 ��ʼ��ÿ�δ����ԲȦ��ɾ����m ������{
http://en.wikipedia.org/wiki/Josephus_problem
==> Approach: 
		--> Q1: N people start from s and count k, what's the index? 
				A1: (s+k-1)mod N if index is from 0 --> N-1; (s+k-2)mod(N-1) +1 if index is from 1 --> N. 
		--> Q2: let f(n,k) be the index of survivor for n counts k, then what's the relation between f(n,k) and f(n-1,k)? �� f(n-1,k) = s-1;
				A2: f(n,k) = ((f(n-1,k) + k) mod n) with f(1,k) = 0 if start from 0; f(n,k) = ((f(n-1,k) + k-1) mod n) + 1 with f(1,k) = 1 if start from 1; 
		--> Solution1 O(n): from A2
		--> Solution2 O(klgn) for k << n: f(n,k) = upper(f(n-lower(n/k))*k/(k-1)) - 1 with f(1,k) = 0 if start 0;  f(n,k) = upper(f(n-lower(n/k))*k/(k-1)) - 1 with f(1,k) = 1 if start 1; 

==> Extension: There are n persons, numbered 1 to n, around a circle. We eliminate second of every k remaining persons until one person remains. Given the n, determine the number of xth person who is eliminated.

==> Same approach: define f(n,k,x), find the relation between f(n,k,x) and f(n-1,k,x-1) down to f(n-x+1,k,1):
		--> f(n,k,1) = (k-1) mod(n) + 1; so f(n-x+1,k,1) = (k-1) mod(n-x+1) + 1;
		--> n peopel start from k+1 and count f(n-1,k,x-1): f(n,k,x) = (k+1+f(n-1,k,x-1)-2)mod(n-1) +1; 
}

N������������{
	ԭ��: 100 ���ܰ�һ�����Ϸֳ������ٿ飿ע�⣬���ܽ��������°ڿ��С�(������ 2^100)
	
	a. ��һά��ʼ˼�����ü�����N������ܽ�һ���߶μ��ɶ��ٶΣ�
	b. �ڶ�ά����£��ü�����N������ܽ���Բ�̼��ɶ��ٿ飿  
	c. ��ά����ԭ�⡣
	d. D ά�أ�
	Let F_D(N) be the maximum number of pieces into which N hyperplanes(cuts) can
	separate a D-dimension ball, where D \geq 1, N \geq 0. Then
		1. F_1(N) = N + 1
		2. F_D(0) == 1
		3. F_D(N) = F_D(N-1) + F_{D-1}(N-1), e.g. F(2,1) = F(2,0) + F(1,0)
	1 and 2 are trivial. while 3 is based on following lemmas.
	Lemma 1. To get the maximum number, the N th cut should intersect with all existing cuts. ��Ϊ��һά���������������������
	Lemma 2. In D-dimension, a hyperplane intersects with K mutually intersected hyperplanes can be separated into F_{D-1}(K) parts. 
	Then the answer of the original question is F_3(100). 
}

��ƽ��������{

1. N ������ ��Ψһһ���������ز�֪���� �� N = 12 ʱ  �������ܽ������ҳ�

==> �ԱȽ�������12��������Ϊ������ƽ��3��״̬�����ÿ�ο��Ի����ϢΪlog3 bit, 12���������ڲ�֪�����أ���˹�12*2 = 24��״̬��2 < log24 / log3 < 3 ���3�ο��Ը㶨�����ҿ����Ƴ�3�������Գ�13���������ַ�����������������������2������(��������������ͬ)����ô�ܹ�C(12,2) * 2 = 132��״̬ ��Ϊ4 < log132 / log3 < 5����ôֻ��Ҫ5�ξ��ܳƳ���Ҳ�����Ƴ�5�������Դ�16����������2�����򡣴���Ϣ�۵õ��Ľ��ֻ��һ������,4 < log132 / log3 < 5 ����һ����ʾ5�ο϶��ܳƳ���,ֻ��˵��4�ο϶��Ʋ�����.

==> ��������һ���������3�ֱ��Σ�
1��n����������һ�����ģ�֪�����ỹ���أ�����ƽ�Ƴ���������
2��n����������һ�����ģ���֪���ỹ���أ�����ƽ�Ƴ���������
3��n����������һ�����ģ���֪���ỹ���أ�����ƽ�Ƴ�������������֪�������ỹ���ء�

��������3�����������n�Σ��������ڼ��������ҳ����������𰸣��ֱ�Ϊ��3^n, (3^n - 1)/2, (3^n - 3)/2.

֤��:
һ����ƽ���أ����������̱Ƚ����أ������������棬Ҳ����ÿ�γ�����3����������� ln3/ln2������Ϣ��n����Ҫ֪������һ����ͬ�������֪���Ǹ���ͬ�����������ỹ���أ��ҳ����Ļ��Ǿ���n������е�һ�֣�������ln(n)/ln2������Ϣ����������Ҫ��k�Σ�������Ϣ���ۣ�k*ln3/ln2>=ln(n)/ln2,  ���k>=ln(n)/ln3 ����Ʒ����ǣ�ÿ���ٴ�����n�����и�ȡ[(n+2)/3]���򣬷�����ƽ����.

��������N(m)=(3^m-1)/2��С������������Ѱ��m�εĽⷨ�� ���ȣ�����m=2��������൱���ĸ�С���������ε����������Ѿ����۹�����ˣ�Ҳ�ܼ򵥣��ڴ���ȥ����Σ���m <=k-1ʱ���ٶ�����N(k-1)=(3^(k-1)-1)/2�����������Ƕ��нⷨ�� ���������� m=k ������� ��һ�γ�ȡ[3^(k-1)-1]���������ƽ��ƽ���ˣ���
      1. ���ƽ�⣬���[3^(k-1)-1]����׼�򣬻�����ʣ�µ�[3^(k-1)+1]/2���С����� [3^(k-1)-1]>=[3^(k-1)+1]/2����k>=2),����֪�ı�׼������С��δ֪����;�������Ժ�Ĳ����о��൱�����������׼����������ǰ����������֪����[3^(k-1)+1]/2�����(k-1)�οɽ⡣
      2. �����ƽ�⣬����Ƿ�����A,С���Ƿ�����B����׼�����C. ������������[3^(k-1)-1]/2��A���B����[3^(k-1)+1]/2��C�� �ڶ�����3^(k-2)��A���[3^(k-2)-1]/2��B������; 3^(k-2)��C���[3^(k-2)-1]/2��A����ұߡ�
       2.1 �����ߴ����ұߣ���˵��������ߵ�3^(k-2)��A�����������Ϊ����
       2.2 �����ߵ����ұߣ���˵�����ڵڶ��γ�ʱû�õ�3^(k-2)��B�����������Ϊ����������������������������ַ�(k-2)�ν��������ǰ���ι�k�ν����2.3 ������С���ұߣ���������ߵ�[3^(k-2)-1]/2��B���л����ұߵ�ͬ����Ŀ��A���С���ʱ������͵ڶ��ο�ʼʱ���ƣ�ֻ������k-1���k-2). ����ͬ�İ취һֱ����׷�ݵ�һ��A��
��һ��B��һ�����ֵ��������ʱֻ����A ��ͱ�׼��Ƚ����¾����ˡ���������������Ҳ�ǿ���������k�ν���ġ�
   ����������������ѧ���ɷ�֪������N(m)=(3^m-1)/2���������m���ǿ��ԳƳ����ġ�

������ⷨ����ǰ�����������Ͻ�Nmax(m) <=(3^m-1)/2��֪��m���ܽ��������С���� Nmax(m)=(3^m-1)/2�� 
����ͬ���֤ n <= (3^k)/2, ���� n ȡ�������� n <= (3^k-1)/2, ��

==>  ͨ���Ե�һ���˼��������ǿ��Է���һЩ���ɣ����ڳ�n�Σ��������ܰѳ�n-1�ε���������n�ξͿ����õ����������ĸ���������������Ҫ�����������⣬��f(n)��ʾ���� ���ǣ�Ҳ���ǵ����򵥵ĵ��ơ���Ϊ�����ڳƵ�һ�ε�ʱ�򣬵õ���һЩ��Ϣ����Щ��Ϣ���Ը����Ǻ���ĳ������������ ������������һ�γƵ�������
  a. ����ƽ�⣬���ǵõ�����Ϣ�ǣ�1�� ����������ϵ������2�� ��һ�ѵ����أ�һ���ᡣ�����������ӵڶ�����Ϣ��ʵ����������Ϣ�Ƿǳ���Ҫ�ġ�������֪��һЩ������ع�ϵ�����ǿ����ñȲ�֪�������ϵ�ƵĴ������پ͵ó����ۡ��磺�������㻵���ᣬ��ô27����ֻҪ���ξ͹��ˡ���������Ҫ�о�һ�£�������֪��һЩ������ع�ϵ��n�������ԳƳ����ٸ��������ú���h(n)��ʾ��
   b. ��ƽ�⣬��õ�����Ϣ�ǣ�1�� ������ʣ�µ�һ���У�2�� �����ɸ�������Ը��������á��ڶ�����Ϣ���Ǵ�����׺��ӵġ�����12���򣬳Ƶ�һ����ƽ�⣬���ǾͿ�������ƽ�ϵ�����Ϊ��׼���б�׼�����ο��ԳƳƳ�4���򣨼���һ�����𲿷ֵ�2��5������û�еĻ�����ֻ�ܳƳ�3�����������ǻ�Ҫ�о�һ�£���������һ����׼��n�������ԳƳ����ٸ��������ú���g(n)��ʾ��
  
����һ����һ������֪����������ƫ�أ���֪��������ƫ�ᣩ�������ǳƴ���Ϊ��ȷ�����򣨻��ȷ�����򣩣���ȷ������Ͱ�ȷ������ͳ��Ϊ��ȷ���� ��һ �� �У�ͨ����һ�γ��غ�����ƽ�⣬��1,2,3,4,5,6,7,8���򶼳�Ϊ��ȷ������1��2��3��4��5��6��7��8����1,2,3,4Ϊ��ȷ������5,6,7,8Ϊ��ȷ������

���������һ������֪�����Ǻ��������ǳƴ���Ϊȷ��������֪���ǻ���ȷ������ȷ�������ȷ������ͳ��Ϊȷ���򡣵�һ���У�ͨ����һ�γ��غ���ƽ �⣬��1,2,3,4,5,6,7,8���򶼳�Ϊȷ�������

����������һ���򣬼Ȳ���ȷ����Ҳ���ǰ�ȷ�������������ǳƴ���Ϊ��ȷ����

��һ���У�ͨ����һ�γ��غ���9��10��11��12���򶼳�Ϊ��ȷ���� һ��δ��֮ǰ���������ǲ�ȷ����
����һ�����ڷ��Ϲ���ƽ���򣬶��ǰ�ȷ�������ȷ���� ���Ǹ���Ȼ���������⡣

�����ģ����������ǰ�ȷ������ôn�οɳƳ������������������ h(n)��ʾ��
�������h(n)=3^n��
֤����
�ù��ɷ���֤��
�Ŷ���n=1,��֤3�����ǿɳƵģ���֤4���ǲ��ɳƵġ�
�� 3����ɳƣ���ȫΪ��ȷ����������������������ƽ�⣬�صľ��ǻ����򣻷���ʣ�µ��Ǹ����ǻ�����ȫΪ��ȷ������ͬ����������ȷ������һ����ȷ�����������������ȷ����������ƽ�⣬�صľ���ȷ�����򣻷���ʣ�µ��Ǹ�����ȷ��������һ����ȷ������������ȷ������ͬ�� ���ԣ�3����ɳơ�
���ĸ��򲻿ɳ�����4������ƽ��һ�Σ�ֻ���ṩ������Ϣ���ɳ���ԭ����Ȼ�����������Ϣ����ͬ�ġ���һ���޷���֤���жϳ����� �ʣ�n=1��h(n)=3^n�ǳ����ġ�
����n = kʱ�������������n=k+1
  ����֤t=3^(k+1)�����ǿ��жϵģ���t����a����ȷ������b����ȷ������,t =a + b ;   �ɶԳ��ԣ�������a>b (a + b�����������Բ��������)    �����·�����Ϊ���ѣ� ��a>=2*(3^k),����ƽ���߸���3^k����ȷ����������ƽ�⣬�������ص��Ƕ��У�ƽ��Ļ���������ʣ�µ��Ƕ��С���ʱʣ3^k����k�ο��жϳ�������k+1�Σ���������a<2*(3^k), ����ƽ���߸���[a/2]����ȷ������,3^k-[a/2]����ȷ����������ƽ�⣬�������ص��Ƕ��еİ�ȷ�����������ǶѰ�ȷ�������У�ƽ��Ļ����������µ��Ƕ��С���ʱʣ3^k����k�ο��жϳ�������k+1�Σ������� a < b ͬ���֤�� ����3^(k+1)�����ǿ��жϵġ�
����3^(k+1)+1���򣬳�һ�Σ�ֻ���ṩ������Ϣ���ɳ���ԭ����Ȼ��3^k+1�������Ϣ����ͬ�ģ���3^k+1�����޷���k�γƳ�����k+1���޷���֤���жϳ����� �ʣ�n=k+1Ҳ������ �ɹ��ɷ���h(n)=3^n��һ����Ȼ�����������ٻص�ԭ�⣬����f(n).���ڵ�һ�δ��������ƽ�⣬��ƽ���ߵ��򶼽���Ϊ��ȷ�����������������Ϊa��������һ�Ѹ���Ϊb������֪��a��b�໥������Ϊʹ��f(n)=2a+b��󣬼�Ҫ�ֱ����a ,b�����ֵ�������������2a����ȷ����Ҫ��n-1���жϳ������ҽ��� 2a <= h (n-1)=3^(n-1). ���Ⱥ��޷�ȡ������Ϊ3^(n-1)Ϊ����,����2a<=3(n-1)-1, max(a)=(3^(n-1)-1)/2  f(n)=(3^(n-1)-1)+max(b)

�����ġ���һ��ȷ������n���ܳƵ�����ȷ����ĸ�����g(n)��ʾ��
��������g(n)=f(n)+1

��������ȷ������Ļ���g(n)��f(n)�����Ч�ʵĵط������ڵ�һ�γƵ�ʱ��������߷ŵķ�ȷ����ĸ�����һ���ࡣһ�߷�c����ȷ����һ�߷�a����ȷ�����c-a��ȷ������ʣ��һ��Ϊb����ƽ��Ļ�,b����Ĵ���ͬf(n),���Դ�ʱ��max(b)��Ȼ����f(n)��max(b)������ƽ��Ļ�������a + c����ȷ���� a + c <= 3 ^ ( n �C 1 )    ���Ⱥſ�ȡ����c=a+1=(3^(n-1)+1)/2,���ʱֻ��Ҫһ��ȷ������, g(n)=max(a + c)+max(b)=3^(n-1)+max(b)=f(n)+1 �����о�max(b).����ƽ��,��b����ȫΪȷ������,����,ȫΪ��ȷ���򡣵���ƽ�ϵ���ȫΪȷ��������ʱ��b��ǡ��ͬ���Ǹո����۵�g(n). ����max(b)=g(n-1)=f(n-1)+1. ����f(n)=3^(n-1)-1+f(n-1)+1 =f(n-1)+3^(n-1) ����һ�����ƹ�ʽ�� ��������֪��f(2)=3 ,�����׽�� f (n)=(3^n-3) / 2 �ʣ���n�Σ���������(3^n-3) / 2�������ҳ���������������ֻ��һ���� ��n=3ʱ������һ�⣬f(3)=12.  3���ܳƵõ������Ŀ����12����

==> ��N����������R������׼��(ͳһ�����������׼��������һ)��������Ϊ��׼�����ֻ�������K�Σ�Ҫ��N�������ҳ���R������׼�����÷ֶκ�����ʾ��N��R��K�Ĺ�ϵ 
}

Given a dictionary and a word, change the word by deleting one letter, output a path in the dictionary to NULL{
	Solution (1): Ϊ dictionary �е� words �� directed graph, ���� DSP �ж���Щ �����ͨ NULL. 
	Solution (2): Top-down DP + DFS, using hash table save unfinished path and judge the accessible of the words. �� FalseSet and ExploredSet ����ѭ��
}

Given dict d and string s, ��ɾ�� string ������ĸ(����), ʹ��ʣ����ĸ����� dict ��� ���ɸ�����{
	e.g. d = {window, cat},  s = "iwndowdcta", ֻɾ��һ����ĸ d �� "window cat"
	==> solution �ö�ά DP ��:
	f_k(i) := s[i,i+k) ɾ�������ٵ�����. Starting from k = 1 initializing f_1(i) = 0 if s[i] is in dict, f_1(i) = 1	otherwise; then calculate f_{k}(i) recursively by:	f_{k+1}(i) = min_{h=1}^{k} {f_h(i) + f_{k+1-h}(i+h)} if s[i,i+k+1) is NOT in dict; otherwise f_{k+1}(i) = 0; up to k = len(s)
	
	--> ����ж� s[i,i+k) is in Dict? One-pass preprocessing
	Sol(1): the signatures of all words in Dict: the frequencies of letters in that word, Ȼ����� signature ����
	Sol(2): Hash table all words
}

n ��ը����һ���ϣ�ÿ���� val �� range(���ߵ�), ����һ��ֻ�ܵõ��õ��� val, ���ò��� range ��ͬʱ������ը��������ȡ�õ���� val sum{
	==> solution: Top down DP(forward DP/ DP with memoization)
	ά��һ�� map<subset,val>: for each subset, �Ȳ��ڸ� map ����û�� value, ���� top-down ����value
		
	double GetVal(vector<int> & val, int Preval, vector<int> &range, set<subset> & path, map<subset,int> & SubsetVal){// b.count > 0
		vector<int> RemainBomb = GetRemainBomb(path.back());
		if(RemainBomb == 1){
			SubsetVal.push_back(path.back(),Preval + val[RemainBomb]);
			return Preval + val[RemainBomb[0]];
		}
		
		int Max = 0;
		for(int i = 0; i < RemainBomb.size(); ++i){
			subset newSet = path.back() - Bomb[RemainBomb[i]];
			it = SubsetVal.find(newSet);
			if(it != SubsetVal.end()){
				Max = max(Max,it->second);
			}else{
				path.push_back(newSet);
				Max = max(Max,GetVal(val,Preval+ val[RemainBomb[i]],range, path,SubsetVal));
				path.pop_back();
			}
		}
	SubsetVal.find(path.back(),Max);
	return Max;
}

2 * n balls (n blue, n red) �������ų�һ�š�Ҫ�������λ����������blue ball����������С��red ball�����������ж������ŷ�{
	==> �� k �� blue ball ����� i = 0, ..., k �� red ball f(k,i) = sum_s^min{i,k-1} f(k-1,s); 
}

�ҳ� array A[n] ��һ�� subset ʹ֮Ԫ�غ� mode n == 0{
	==> solution: ���� max sub array sum ��� p(j) := \sum^j_0 A[j] mode n. �� p(i) i = 0, ..., n-1 �����ֿ���: 1. ���ʣ�������һ��Ϊ 0; 2. ������������ͬ, ��Ϊ�����Ӽ�
}

Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses{
	==> wrong solution: DP f(n) = (f(n-1)) and ()(f(n-1)) and [(f(n-1))() if f(n-1)[0:1] != ()], ע�� replicates ==> ���� �� n == 4 ʱ��û�а���(())(())
	==> solution: ���͵ĵݹ顣һ���������ַ������������ų��ִ���<nʱ���Ϳ��Է����µ������š��������ų��ִ���С�������ų��ִ���ʱ���Ϳ��Է����µ������š�		
	    void GenerateParenthesis_sub(vector<string> &result, string path, int n, int LeftNum){
        if(LeftNum == n){// no more '('
            int AddRightNum = n + n - path.size();
            while(AddRightNum-- > 0) path.push_back(')');
            result.push_back(path);
            return;
        }
        if(LeftNum > path.size()/2){// valid to add ')'
            path.push_back(')');
            GenerateParenthesis_sub(result,path,n,LeftNum);
            path.pop_back();
        }
        path.push_back('(');
        GenerateParenthesis_sub(result,path,n,LeftNum+1);
    }
    vector<string> generateParenthesis(int n) {
        // Note: The Solution object is instantiated only once and is reused by each test case.
        vector<string> result;
        string path;
        GenerateParenthesis_sub(result,path,n,0);
        return result;
    }
  ==> Extension: ��n1��()��n2��[]��n3��{}��ö�ٳ����еĺϷ�������ϡ�ע�ⲻ����Ϸ������Ÿ���

}

n dice * m sides with equal prob. given n,m,x; calculate the probability that the score is greater than or equal to x{
	http://www.careercup.com/page?pid=google-interview-questions
	
	For eg:
	n=4 m=2 x=8
	ans: 1/8=0.125
	n=2 m=6 x=3
	ans:35/36=0.972
	
	==> Solution:  Using forward DP, O( n*min(n*m,x)): p(n,x) the probability with n dices, then "p(1,k) == 1/m, k = 1,..m" and "p(i, x) = \sum_{i = 1}^m p(i-1, x-k) / m" for i = 2,...,n
}

Given a dictionary of millions of words, find the largest possible rectangle of letters such that every row forms a word (reading left to right) and every column forms a word (reading top to bottom){// CareerCup 11.8 ??? } 

Given a m*n matrix, each element has a value, ֻ�ܴ�ֵ��������ֵ���ߣ����·��{// ע�ⲻ��ֻ�����µ��ߣ����ܼ򵥵��� DP, 
	==> ˼������ʵ ֵ�ĸߵ��Ѿ���Ϊ topological sort�� finish time ע���·�����������СԪ��, 
	==> solution: 
		1. ��Ԫ�� dist = 0. 
		2. ������С�� heap, ���� neighbor ��ֵ��С������ heap, d[i] = max{d[i], d[pre]+1}
}

�� k �� ������������ G_1(n), ... G_k(n) ��һ���� s, ����䷽�� [a_1 ... a_k ] s.t. \sum {a_k} G_k(a_k) ��� ������ \sum {k} a_k = s {// O(s^2*k)
	==> f([a_1 ... a_k ]_m) Ϊ s == m ʱ�����ֵ ����䷽�� [a_1 ... a_k ]_m, ��ݹ鹫ʽΪ 
	f([a_1 ... a_k ]_m) = max{ f(max[a_1 ... a_k ]_{m-t})}, t = 1,...,m-1
	��ֵ����: f([]_1) = max_k G_k(1) 
}

// ================================================================= //
��������Ž�  2012.10.09

==> 0/1 ����: N ����Ʒ (cost c_i, value v_i), ������ C, �� subset s.t. \sum v_i ��������� \sum c_i <= C
	--> solution O(CN) ÿ��һ����Ʒ��һ��ѭ��: f_i(c) Ϊ ��ǰ i ����Ʒװ�� ����Ϊ c �İ��������ֵ���� f_i(c) = max {f_{i-1}(c), f_{i-1}(c-c_i) + v_i} "for any c <= C", 
	--> ע�� (1) i = 1 ... N; �� c �Ӻ�ͷ C ... 0, ������ظ�����; (2) ע�⵽ f_i �ĸ���ֻ�õ� f_{i-1} �� c ǰ(���� c) ����Ϣ��������һ�� "size == c" ��һά���� 

==> ÿ����Ʒ�ĸ����ϵ�����
	--> ��ȫ����: N "��"��Ʒ, �� 0/1 ������ͬ�ĵط����� ����ѡȡ���
	--> solution ���Ӷ����: �� 0/1 ������"ȡ"��"��ȡ" �� "ȡ���ٸ�" f_i(c) = max {f_{i-1}(c-k*c_i) + k* v_i} 	
	--> ���������С��Ŀ�����: �ɹ���ÿ����Ʒ�Ķ����Ʊ�ʾ  v_i * 2^k, c_i * 2^k, k  up to floor(C/c_i)
	--> ���ر���: ����ȫ������һ������ ÿ����Ʒ��಻���� K ��
	--> ��ϱ���: ��Ʒ���Էֱ��� 0/1, ��ȫ, ����. Solution: f_i(c) �ĸ��¹�ʽ��"ÿ��"��Ʒȷ��

==> �������ı���: ���ڹ�����"���", 
	--> ���鱳��: ��Ʒ�ֳ� K �飬ÿ��ֻ��ѡһ��; ��������ǰ��� update f_k(c) = max {f_{k-1}(c), max_in_k {f_{k-1}(c-c_i)+ v_i } }
	--> �������: ÿ�������ͬһ��. e.g. �����Ƿ��ң����: {��һ���ӣ����Ӽ�ң����} Ϊһ��
		
==> ��ά����: �� cost ��һ��ά�ȣ�f_i(c1, c2) ����
	
==> value �ı���:
	--> ����СӲ���������, v = Ӳ�Ҹ�����min Ӳ�Ҹ���
	--> �����Ӳ���������, v = abs(��ϵĲ�), 
	--> ��װ���������ܷ�����.e.g. ���������Ϸ�����Ŀ.  f_i(c) = f_{i-1}(c) + f_{i-1}(c-c_i)
	--> ����������ֵ����������ɱ�ʾ����Ŀ�ĸ���: �� bool 
	--> ���� N �� array, ������������which �ܹ���ÿ��array���Ԫ�����(sum) ����
	--> ����һ��������ÿ��Ԫ�ص���Ҫ����һ�� subset ��, ��� subset �е�Ԫ��
	--> ʹ���������ʣ��ռ���С f_i(c) ����Ϊʣ��Ŀռ�;  max ��� min

==> ��� K �� (��ǰ K ��) ���ŷ���, ����ǰ K �������� update f_{i,k} ʱ��max{f_(i-1,k)(c), f_(i-1,k)(c-c_i)+v_i} ������������ O(KCN)
// ================================================================= //

��̬���򾭵�̳�: 
==> DP ǰ��: overlapping subproblem + optimal structure	

==> ����(һά): 
			* ��ǽ�(����)������: p(i) Ϊ�� s[i] Ϊ��������г���. 
			* ��С��Ŀ����������: ת������������еĳ���. 
			* ��������Ҫ���һԪ�ر�ǰһԪ�����ٴ� k: ת���������(ǰһԪ�رȺ�һԪ������k-1)�ĳ���
			* ���������������: ���Ҵ�ͷ��β���� p(i)�� ���Ҵ�β��ͷ���½� p(i), �������� max_sum. ������������� stock ��˼��һ�� 
			* �� A[] �ﰴ B[] ������ֵ������������, e.g. : ����ཻ��������
		
==> ��ά DP����ά����, (���, ����), (���, �յ�), (x,y ����), (��1���꣬��2����), (��һ·��ÿ i �����꣬�ڶ�·��ÿ i �����꣬)
			* ���������, 
			* ���Ĵ� (���� int ���� bool �ﵽ��ά)
			* Edit Distance
			* (����)���кϲ� (�� value �� sum or product)
			* ��һ dictionary �� string, ��ָ� string �� K ����, ʹ��ÿ����(��ͬһ�Ӵ����ĸ��ǰһ��������ĸǰ����ĸ�ⶼ���������´���)�ܹ��ɵĵ����������: cnt_k(start,len) k = 1,...,K; start = 0,...,n-1; len = n - start
			* �����ֲ��������ʹ�ý�����ϸ���Ҫ��: p(i,num) ǰ i �������ʹ�� ���Ϊ num

==> ����
			* ���������ڵ�����ÿ�ڵ�� value, ��ʹ�ü�������õ��� value ���: p(root,k) �� root Ϊ���ķ�֧���� k �� nodes ����� value
			* �����ε�����Ŀ K�����ѡ�޿�ѧ��: ѡ�޿� == �ڵ�, ���� == ���ڵ�. ÿ���ӽڵ���� 1,..., K ��Ŀʱ�����ֵ. �ϲ��ӽڵ�ʱ(������ heap)������ k �� ������������ G_1(n), ... G_k(n) 

// ================================================================= //
�㷨�������ž���+������

==> DAG (Directed Acyclic Graph) �ϵĶ�̬�滮: 
	* ��ʾ��Ԫ�����ϵ: ����Ŀ�Ƕ��, ��������
	* DAG �е��·�� d(i) = max_{(i,j)} {d(j)+1} + memorization				
	* memorization ��Ҫע�� ����û�м���� �� �޽�

==> �Ӽ�
	* 2K ��Ԫ��������ԣ�ʹ֮ÿ�Եľ������С

