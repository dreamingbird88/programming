// ================================================================= //
阅读记录: 2013.10.04 2013.10.24
// ================================================================= //

四边形不等式 --> DP 加速

一个整数数组，长度为n，将其分为m份，使各份的和相等，求m的最大值{// 用 DP ???
  比如{3，2，4，3，6} 可以分成{3，2，4，3，6} m=1;
  {3,6}{2,4,3} m=2
  {3,3}{2,4}{6} m=3 所以m的最大值为3
  
	==> solution:
		1. 求 Max 及 Sum 并将 sum 因式分解。其中 Max 应为因子的下界, 即 m <= Sum/Max; 分解 F[i] with A[i] 次方数, i = 0,...,f-1
		2. 用 merge f-sort array 到 queue 的做法从小到大得到分解因子，每个因子对应分解后每组之和，对于 < Max 忽略
		3. 数组排序 O(nlgn), 对每个因子，问题变成
}

Given an array，分成两份，使两份之和尽量相等{// 背包问题 
	==> solution 0: Backpack problem f_i(x) = min{f_{i-1}(x), f_{i-1}(x-a[i]), |x-a[i]|} where x is sum/2; 实现有问题
	==> solution 2: 先建立 长度为 sum / 2 的 bool 型 (或者 set 有排序功能)，从大到小排列元素(好像多余)，对每个 a[i], 检查 set 中的 sum s有没有 可以 s + a[i] == sum/2, 若有则 return true
	==> Extension: 使得两份数量相等 f_i(x,n) = min{f_{i-1}(x,n), f_{i-1}(x-a[i],n-1), |x-a[i]| if n == 1} with n is num/2;
}

给出总数 x 与货币各种面值 {k1, k2, ..., km}，求{// 背包问题 
	==> 最优货币结合方案，使得与总数的差值最小 e.g. $417 50c/10c/7c 
	--> Solution: p_m[i] = min{min{p_{m-1}[i-k_m * s]}, V%k_m}, i = 0,...,x the least overpaid value
	==> 用最少货币组成 sum: 用sum 对货币二进制化，再用 01 背包 f_{i}(s) {k1, k2, ..., ki} 组成 sum == s 的最小数 注意应从最小货币面额开始
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

Josephus problem n 个数字（0,1,…,n-1）形成一个圆圈，从数字0 开始，每次从这个圆圈中删除第m 个数字{
http://en.wikipedia.org/wiki/Josephus_problem
==> Approach: 
		--> Q1: N people start from s and count k, what's the index? 
				A1: (s+k-1)mod N if index is from 0 --> N-1; (s+k-2)mod(N-1) +1 if index is from 1 --> N. 
		--> Q2: let f(n,k) be the index of survivor for n counts k, then what's the relation between f(n,k) and f(n-1,k)? 即 f(n-1,k) = s-1;
				A2: f(n,k) = ((f(n-1,k) + k) mod n) with f(1,k) = 0 if start from 0; f(n,k) = ((f(n-1,k) + k-1) mod n) + 1 with f(1,k) = 1 if start from 1; 
		--> Solution1 O(n): from A2
		--> Solution2 O(klgn) for k << n: f(n,k) = upper(f(n-lower(n/k))*k/(k-1)) - 1 with f(1,k) = 0 if start 0;  f(n,k) = upper(f(n-lower(n/k))*k/(k-1)) - 1 with f(1,k) = 1 if start 1; 

==> Extension: There are n persons, numbered 1 to n, around a circle. We eliminate second of every k remaining persons until one person remains. Given the n, determine the number of xth person who is eliminated.

==> Same approach: define f(n,k,x), find the relation between f(n,k,x) and f(n-1,k,x-1) down to f(n-x+1,k,1):
		--> f(n,k,1) = (k-1) mod(n) + 1; so f(n-x+1,k,1) = (k-1) mod(n-x+1) + 1;
		--> n peopel start from k+1 and count f(n-1,k,x-1): f(n,k,x) = (k+1+f(n-1,k,x-1)-2)mod(n-1) +1; 
}

N刀切西瓜问题{
	原题: 100 刀能把一个西瓜分成最多多少块？注意，不能将西瓜重新摆开切。(即不是 2^100)
	
	a. 从一维开始思考：用剪刀剪N次最多能将一段线段剪成多少段？
	b. 在二维情况下：用剪刀剪N次最多能将个圆盘剪成多少块？  
	c. 三维就是原题。
	d. D 维呢？
	Let F_D(N) be the maximum number of pieces into which N hyperplanes(cuts) can
	separate a D-dimension ball, where D \geq 1, N \geq 0. Then
		1. F_1(N) = N + 1
		2. F_D(0) == 1
		3. F_D(N) = F_D(N-1) + F_{D-1}(N-1), e.g. F(2,1) = F(2,0) + F(1,0)
	1 and 2 are trivial. while 3 is based on following lemmas.
	Lemma 1. To get the maximum number, the N th cut should intersect with all existing cuts. 因为多一维，所以这个条件可以满足
	Lemma 2. In D-dimension, a hyperplane intersects with K mutually intersected hyperplanes can be separated into F_{D-1}(K) parts. 
	Then the answer of the original question is F_3(100). 
}

天平称球问题{

1. N 个球里 有唯一一个坏球，轻重不知，求 当 N = 12 时  称三次能将该球找出

==> 以比较有名的12彩球问题为例，天平有3种状态，因此每次可以获得信息为log3 bit, 12个彩球，由于不知其轻重，因此共12*2 = 24种状态，2 < log24 / log3 < 3 因此3次可以搞定。并且可以推出3次最多可以称13个球。用这种方法还可以推算出，假如存在2个坏球(两个坏球质量相同)，那么总共C(12,2) * 2 = 132种状态 因为4 < log132 / log3 < 5，那么只需要5次就能称出，也可以推出5次最多可以从16个球中挑出2个坏球。从信息论得到的结果只是一个下限,4 < log132 / log3 < 5 并不一定表示5次肯定能称出来,只是说明4次肯定称不出来.

==> 称球问题一般会有以下3种变形：
1、n个球，其中有一个坏的，知道是轻还是重，用天平称出坏球来。
2、n个球，其中有一个坏的，不知是轻还是重，用天平称出坏球来。
3、n个球，其中有一个坏的，不知是轻还是重，用天平称出坏球来，并告知坏球是轻还是重。

对于上面3种情况，称量n次，最多可以在几个球中找出坏球来？答案：分别为：3^n, (3^n - 1)/2, (3^n - 3)/2.

证明:
一、天平称重，有两个托盘比较轻重，加上托盘外面，也就是每次称重有3个结果，就是 ln3/ln2比特信息。n个球要知道其中一个不同的球，如果知道那个不同重量的球是轻还是重，找出来的话那就是n个结果中的一种，就是有ln(n)/ln2比特信息，假设我们要称k次，根据信息理论：k*ln3/ln2>=ln(n)/ln2,  解得k>=ln(n)/ln3 具体称法就是：每次再待定的n个球中各取[(n+2)/3]个球，放在天平两边.

二、对于N(m)=(3^m-1)/2个小球，现在我们来寻求m次的解法。 首先，对于m=2的情况，相当于四个小球来称两次的情况，这个已经讨论过多次了，也很简单，在此略去。其次，若m <=k-1时，假定对于N(k-1)=(3^(k-1)-1)/2个球的情况我们都有解法。 现在来考虑 m=k 的情况。 第一次称取[3^(k-1)-1]个球放在天平天平两端，则：
      1. 如果平衡，获得[3^(k-1)-1]个标准球，坏球在剩下的[3^(k-1)+1]/2个中。由于 [3^(k-1)-1]>=[3^(k-1)+1]/2，（k>=2),即已知的标准球数不小于未知球数;所以在以后的测量中就相当于任意给定标准球的情况，由前面的引理二可知对于[3^(k-1)+1]/2的情况(k-1)次可解。
      2. 如果不平衡，大的那方记做A,小的那方记作B。标准球记做C. 则现在我们有[3^(k-1)-1]/2个A球和B球，有[3^(k-1)+1]/2个C球。 第二次用3^(k-2)个A球加[3^(k-2)-1]/2个B球放左边; 3^(k-2)个C球加[3^(k-2)-1]/2个A球放右边。
       2.1 如果左边大于右边，则说明是在左边的3^(k-2)个A球中质量大的为坏球；
       2.2 如果左边等于右边，则说明是在第二次称时没用的3^(k-2)个B球中质量轻的为坏球。以上两种情况都可以再用三分法(k-2)次解决，加上前两次共k次解决。2.3 如果左边小于右边，则坏球在左边的[3^(k-2)-1]/2个B球中或在右边的同样数目的A球中。此时的情况和第二次开始时类似（只不过是k-1变成k-2). 用相同的办法一直往下追溯到一个A球
和一个B球一次区分的情况，这时只需拿A 球和标准球比较以下就行了。因此在这种情况下也是可以最终用k次解决的。
   由以上两步加上数学归纳法知，对于N(m)=(3^m-1)/2的情况，称m次是可以称出来的。

由这个解法加上前面所给出的上界Nmax(m) <=(3^m-1)/2，知称m次能解决的最大的小球数 Nmax(m)=(3^m-1)/2。 
三、同理可证 n <= (3^k)/2, 由于 n 取整，所以 n <= (3^k-1)/2, 而

==>  通过对第一题的思索解答，我们可以发现一些规律：对于称n次，若我们能把称n-1次的问题解决，n次就可以用递推来求出球的个数。（对于我们要解决的这个问题，用f(n)表示。） 但是，也不是单纯简单的递推。因为我们在称第一次的时候，得到了一些信息，这些信息可以给我们后面的称球带来帮助。 我们来分析第一次称的三堆球：
  a. 若不平衡，我们得到的信息是：1． 坏球在天边上的两堆里；2． 有一堆的球重，一堆轻。大家往往会忽视第二条信息，实际上这条信息是非常重要的。若我们知道一些球的轻重关系，我们可以用比不知道这个关系称的次数更少就得出结论。如：若告诉你坏球轻，那么27个球只要三次就够了。所以我们要研究一下，若我们知道一些球的轻重关系，n次最多可以称出多少个球。我们用函数h(n)表示。
   b. 若平衡，则得到的信息是：1． 坏球在剩下的一堆中；2． 有若干个好球可以给我们利用。第二条信息又是大家容易忽视的。就如12个球，称第一次若平衡，我们就可以用天平上的球作为标准球。有标准球，两次可以称称出4个球（见第一试题解答部分的2—5）；若没有的话，就只能称出3个球。所以我们还要研究一下，若我们有一个标准球，n次最多可以称出多少个球。我们用函数g(n)表示。
  
定义一：若一个球，若知道它不可能偏重（或知道不可能偏轻），则我们称此球为半确定重球（或半确定轻球）；半确定重球和半确定轻球统称为半确定球。 第一 题 中，通过第一次称重后，若不平衡，则1,2,3,4,5,6,7,8号球都成为半确定球，若1，2，3，4〈5，6，7，8，则1,2,3,4为半确定轻球，5,6,7,8为半确定重球。

定义二：若一个球，若知道它是好球，则我们称此球为确定好球；若知道是坏球，确定坏球。确定好球和确定好球统称为确定球。第一题中，通过第一次称重后，若平 衡，则1,2,3,4,5,6,7,8号球都成为确定球好球。

定义三：若一个球，既不是确定球，也不是半确定球若，则我们称此球为不确定球。

第一题中，通过第一次称重后，则9，10，11，12号球都成为不确定球。 一次未称之前，所有球都是不确定球。
引理一、对于放上过天平的球，都是半确定球或是确定球 这是个显然成立的命题。

定义四：若所有球都是半确定球，那么n次可称出的球的最大个数我们用 h(n)表示。
引理二：h(n)=3^n。
证明：
用归纳法来证：
⑴对于n=1,先证3个球是可称的，再证4个是不可称的。
① 3个球可称，若全为半确定重球，任意挑两个，若不平衡，重的就是坏重球；否则，剩下的那个就是坏重球；全为半确定轻球同理；若两个半确定重球，一个半确定轻球，则称两个两半确定重球，若不平衡，重的就是确定重球；否则，剩下的那个就是确定轻球；若一个半确定重球，两个半确定轻球同理。 所以，3个求可称。
②四个球不可称若是4个球，天平称一次，只能提供三条信息，由抽屉原理，必然有两个球的信息是相同的。故一次无法保证能判断出来。 故，n=1是h(n)=3^n是成立的。
⑵设n = k时命题成立，对于n=k+1
  ①先证t=3^(k+1)个球是可判断的：设t中有a个半确定重球，b个半确定轻球,t =a + b ;   由对称性，不妨设a>b (a + b是奇数，所以不可能相等)    按如下方法分为三堆： 若a>=2*(3^k),则天平两边各放3^k个半确定重球。若不平衡，坏球在重的那堆中；平衡的话，坏球在剩下的那堆中。这时剩3^k个球，k次可判断出来，共k+1次，成立。若a<2*(3^k), 则天平两边各放[a/2]个半确定重球,3^k-[a/2]个半确定轻球。若不平衡，坏球在重的那堆中的半确定重球或轻的那堆半确定轻球中；平衡的话，坏球在下的那堆中。这时剩3^k个球，k次可判断出来，共k+1次，成立。 a < b 同理可证。 所以3^(k+1)个球是可判断的。
②若3^(k+1)+1个球，称一次，只能提供三条信息，由抽屉原理，必然有3^k+1个球的信息是相同的，这3^k+1个球无法用k次称出。故k+1次无法保证能判断出来。 故，n=k+1也成立。 由归纳法，h(n)=3^n对一切自然数都成立。再回到原题，来求f(n).对于第一次处理后，若不平衡，天平两边的球都将成为半确定球。设两边球个数各为a个，另外一堆个数为b个。易知，a与b相互独立。为使得f(n)=2a+b最大，即要分别求出a ,b的最大值。由引理二，这2a个半确定球要在n-1次判断出，当且仅当 2a <= h (n-1)=3^(n-1). 但等号无法取到，因为3^(n-1)为奇数,所以2a<=3(n-1)-1, max(a)=(3^(n-1)-1)/2  f(n)=(3^(n-1)-1)+max(b)

定义四、给一个确定好球，n次能称的最多非确定球的个数用g(n)表示。
引理三：g(n)=f(n)+1

若有任意确定好球的话，g(n)比f(n)可提高效率的地方就在于第一次称的时候可以两边放的非确定球的个数不一样多。一边放c个不确定球，一边放a个不确定球和c-a个确定好球，剩下一堆为b个。平衡的话,b个球的处理同f(n),所以此时的max(b)显然等于f(n)的max(b)。若不平衡的话，就有a + c个不确定球。 a + c <= 3 ^ ( n – 1 )    （等号可取）令c=a+1=(3^(n-1)+1)/2,则此时只需要一个确定好球, g(n)=max(a + c)+max(b)=3^(n-1)+max(b)=f(n)+1 再来研究max(b).若不平衡,则b个球全为确定好球,否则,全为非确定球。但天平上的球全为确定好球。这时的b就恰好同我们刚刚讨论的g(n). 即有max(b)=g(n-1)=f(n-1)+1. 故有f(n)=3^(n-1)-1+f(n-1)+1 =f(n-1)+3^(n-1) 这是一个递推公式。 我们又易知，f(2)=3 ,所以易解得 f (n)=(3^n-3) / 2 故，称n次，最多可以在(3^n-3) / 2个球中找出坏球来。（坏球只有一个） 当n=3时，即第一题，f(3)=12.  3次能称得的最大数目将是12个。

==> 有N个球，其中有R个不标准球(统一重量，但与标准球重量不一)，其他均为标准球，如果只允许称重K次，要从N个球中找出这R个不标准球，试用分段函数表示出N、R、K的关系 
}

Given a dictionary and a word, change the word by deleting one letter, output a path in the dictionary to NULL{
	Solution (1): 为 dictionary 中的 words 建 directed graph, 再用 DSP 判断那些 点可以通 NULL. 
	Solution (2): Top-down DP + DFS, using hash table save unfinished path and judge the accessible of the words. 分 FalseSet and ExploredSet 避免循环
}

Given dict d and string s, 求删除 string 最少字母(分离), 使得剩下字母可组成 dict 里的 若干个单词{
	e.g. d = {window, cat},  s = "iwndowdcta", 只删除一个字母 d 得 "window cat"
	==> solution 用二维 DP 做:
	f_k(i) := s[i,i+k) 删除的最少单词数. Starting from k = 1 initializing f_1(i) = 0 if s[i] is in dict, f_1(i) = 1	otherwise; then calculate f_{k}(i) recursively by:	f_{k+1}(i) = min_{h=1}^{k} {f_h(i) + f_{k+1-h}(i+h)} if s[i,i+k+1) is NOT in dict; otherwise f_{k+1}(i) = 0; up to k = len(s)
	
	--> 如何判断 s[i,i+k) is in Dict? One-pass preprocessing
	Sol(1): the signatures of all words in Dict: the frequencies of letters in that word, 然后按这个 signature 排序。
	Sol(2): Hash table all words
}

n 个炸弹在一环上，每个有 val 和 range(两边的), 引爆一个只能得到该弹的 val, 而得不到 range 内同时引爆的炸弹，求能取得的最大 val sum{
	==> solution: Top down DP(forward DP/ DP with memoization)
	维护一个 map<subset,val>: for each subset, 先查在该 map 里有没有 value, 否则 top-down 计算value
		
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

2 * n balls (n blue, n red) 从左到右排成一排。要求从任意位置向左数，blue ball的数量不得小于red ball的数量。求共有多少种排法{
	==> 第 k 个 blue ball 左边有 i = 0, ..., k 个 red ball f(k,i) = sum_s^min{i,k-1} f(k-1,s); 
}

找出 array A[n] 中一个 subset 使之元素和 mode n == 0{
	==> solution: 与求 max sub array sum 差不多 p(j) := \sum^j_0 A[j] mode n. 则 p(i) i = 0, ..., n-1 有两种可能: 1. 互质，则至少一个为 0; 2. 至少有两个相同, 则差集为所求子集
}

Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses{
	==> wrong solution: DP f(n) = (f(n-1)) and ()(f(n-1)) and [(f(n-1))() if f(n-1)[0:1] != ()], 注意 replicates ==> 错误 当 n == 4 时，没有包括(())(())
	==> solution: 典型的递归。一步步构造字符串。当左括号出现次数<n时，就可以放置新的左括号。当右括号出现次数小于左括号出现次数时，就可以放置新的右括号。		
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
  ==> Extension: 有n1个()，n2个[]，n3个{}，枚举出所有的合法括号组合。注意不是求合法的括号个数

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

Given a m*n matrix, each element has a value, 只能从值高者走向值低者，求最长路径{// 注意不是只能右下地走，不能简单地做 DP, 
	==> 思考，其实 值的高低已经成为 topological sort的 finish time 注意最长路径必须包含最小元素, 
	==> solution: 
		1. 将元素 dist = 0. 
		2. 所有最小入 heap, 令其 neighbor 按值从小到大入 heap, d[i] = max{d[i], d[pre]+1}
}

给 k 个 单调递增函数 G_1(n), ... G_k(n) 及一整数 s, 求分配方案 [a_1 ... a_k ] s.t. \sum {a_k} G_k(a_k) 最大 且满足 \sum {k} a_k = s {// O(s^2*k)
	==> f([a_1 ... a_k ]_m) 为 s == m 时的最大值 其分配方案 [a_1 ... a_k ]_m, 则递归公式为 
	f([a_1 ... a_k ]_m) = max{ f(max[a_1 ... a_k ]_{m-t})}, t = 1,...,m-1
	边值条件: f([]_1) = max_k G_k(1) 
}

// ================================================================= //
背包问题九讲  2012.10.09

==> 0/1 背包: N 件物品 (cost c_i, value v_i), 总容量 C, 求 subset s.t. \sum v_i 最大且满足 \sum c_i <= C
	--> solution O(CN) 每加一件物品是一个循环: f_i(c) 为 将前 i 件物品装入 容量为 c 的包里的最大价值，则 f_i(c) = max {f_{i-1}(c), f_{i-1}(c-c_i) + v_i} "for any c <= C", 
	--> 注意 (1) i = 1 ... N; 而 c 从后到头 C ... 0, 否则会重复计算; (2) 注意到 f_i 的更新只用到 f_{i-1} 中 c 前(包括 c) 的信息，可以用一个 "size == c" 的一维数组 

==> 每种物品的个数上的限制
	--> 完全背包: N "种"物品, 与 0/1 背包不同的地方在于 可以选取多件
	--> solution 复杂度提高: 从 0/1 背包的"取"与"不取" 到 "取多少个" f_i(c) = max {f_{i-1}(c-k*c_i) + k* v_i} 	
	--> 如果是求最小数目的组合: 可构造每种物品的二进制表示  v_i * 2^k, c_i * 2^k, k  up to floor(C/c_i)
	--> 多重背包: 比完全背包多一种限制 每种物品最多不超过 K 件
	--> 混合背包: 物品可以分别是 0/1, 完全, 多重. Solution: f_i(c) 的更新公式按"每种"物品确定

==> 有依赖的背包: 在于构造新"物件", 
	--> 分组背包: 物品分成 K 组，每组只能选一件; 按组而不是按件 update f_k(c) = max {f_{k-1}(c), max_in_k {f_{k-1}(c-c_i)+ v_i } }
	--> 主从组合: 每种组合在同一组. e.g. 电视是否加遥控器: {单一电视，电视加遥控器} 为一组
		
==> 多维背包: 即 cost 多一个维度，f_i(c1, c2) 更新
	
==> value 的变型:
	--> 求最小硬币数的组合, v = 硬币个数，min 硬币个数
	--> 求最靠近硬币数的组合, v = abs(组合的差), 
	--> 求装满背包的总方案数.e.g. 法码称重组合方案数目.  f_i(c) = f_{i-1}(c) + f_{i-1}(c-c_i)
	--> 给出货币面值与数量，求可表示的数目的个数: 用 bool 
	--> 给出 N 列 array, 求用最大的数，which 能够被每列array里的元素组合(sum) 出来
	--> 给出一个集合里每个元素的重要，及一个 subset 和, 求该 subset 中的元素
	--> 使个背包里的剩余空间最小 f_i(c) 定义为剩余的空间;  max 变成 min

==> 求第 K 个 (或前 K 个) 最优方案, 保存前 K 个方案， update f_{i,k} 时，max{f_(i-1,k)(c), f_(i-1,k)(c-c_i)+v_i} 两个序列排序 O(KCN)
// ================================================================= //

动态规则经典教程: 
==> DP 前提: overlapping subproblem + optimal structure	

==> 区间(一维): 
			* 最长非降(上升)子序列: p(i) 为以 s[i] 为结点的最长序列长度. 
			* 最小数目上升子序列: 转化成最长非升序列的长度. 
			* 若子序列要求后一元素比前一元素至少大 k: 转化成最长序列(前一元素比后一元素最多大k-1)的长度
			* 求最长先升后降子序列: 先找从头到尾上升 p(i)， 再找从尾到头的下降 p(i), 计算两者 max_sum. 与最多买卖两次 stock 的思想一样 
			* 求 A[] 里按 B[] 次序出现的最长上升子序列, e.g. : 最大不相交航线问题
		
==> 多维 DP：多维背包, (起点, 长度), (起点, 终点), (x,y 坐标), (串1坐标，串2坐标), (第一路径每 i 步坐标，第二路径每 i 步坐标，)
			* 最长公共序列, 
			* 回文词 (可用 int 代替 bool 达到降维)
			* Edit Distance
			* (环型)序列合并 (其 value 可 sum or product)
			* 给一 dictionary 及 string, 求分割 string 成 K 部分, 使得每部分(在同一子串里，字母除前一单词首字母前的字母外都可以用在新词上)能构成的单词数和最大: cnt_k(start,len) k = 1,...,K; start = 0,...,n-1; len = n - start
			* 给数字插入运算符使得结果符合给定要求: p(i,num) 前 i 个数组合使得 结果为 num

==> 树型
			* 给定保留节点数及每节点的 value, 求使得剪割后所得的树 value 最大: p(root,k) 以 root 为根的分支保留 k 个 nodes 的最大 value
			* 给定课的总数目 K，最大化选修课学分: 选修课 == 节点, 基础 == 父节点. 每个子节点计算 1,..., K 数目时的最大值. 合并子节点时(不能用 heap)用上面 k 个 单调递增函数 G_1(n), ... G_k(n) 

// ================================================================= //
算法竞赛入门经典+完整版

==> DAG (Directed Acyclic Graph) 上的动态规划: 
	* 表示二元依存关系: 矩阵的可嵌套, 背包问题
	* DAG 中的最长路径 d(i) = max_{(i,j)} {d(j)+1} + memorization				
	* memorization 中要注意 区别没有计算过 与 无解

==> 子集
	* 2K 个元素两两配对，使之每对的距离和最小

