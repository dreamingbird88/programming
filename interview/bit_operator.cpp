// ====================================================================== //
// Read Record 2013.10.23
// ====================================================================== //

// Swap odd bits with even bits in integer. E.g. swap bit 0 and bit 1, bit 2
// and bit 3: 10001010 --> 01000101
solution: 用 0xaaaaaa 0x555555 找出奇偶位 移位 再用 | 相并

// 最低位 1 清零
solution: i & (i - 1)
  
// 取最低位 1
solution: i & !(i-1);  

// Build a mask, s.t. [i,j] bits == 1, while other bits == 0.
// If ~mask, then [i, j] == 0, while other bits == 1.
solution: (1) [j-i,0] 全为 1: (1<<(j-i+1) - 1);
          (2) 左移 i 位: Mask = (1 << (j-i+1) - 1) << i

// Check if 'n' is a power of 2. I.e. n = 2, 4, 8, ...
solution: 即查是否只含一个 1 	((n & (n-1)) == 0)
// If 'n' is a power of 4.
solutnon: (n & 0xAAAAAAAA) == 0 && (n&(n-1)) == 0)
// If 'n' is a power of 8.
solutnon: (n & 0xcccccccc) == 0 && (n&(n-1)) == 0)
// If 'n' is a power of 16.
solutnon: (n & 0x11111111) == 0 && (n&(n-1)) == 0 && n > 15)
	
// max(A, B)
solution: (1) Perform operation C = (A-B)
          (2) k = Most significant bit of C , i.e k =1 if B>A else K=0
          (3) return A - k*C. This will return the maximum of A and B.

// swap a number in place without temporary variables.
solution: Note 0^1^0 = 1, 1^0^1 = 0, 1^1^1 = 1, 0^0^0 = 0; then x^y^x = y.
So a = a^b; b = a^b; a = b^a;

// use random[1,5] to generate random[1,7]
solution: 即用两位 5 进制数表示 1-7.
int num = 5*(rand5()-1) + rand5() - 1; if (num < 21) return num % 7;
// can use multiple bits of "5" to represent multiple bits of "7"


// In an int array, only one number appear once, others are twice. Find that
// number. 要求遍历一次完成
solution: 用 bit XOR (^) 去对消出现两次的数
// Extension: 若其余数出现三次?
solution: 用一数组 count 每位上的 bit, 再 mod 3:  ones 表示出现一次的, twos 表示出现两次的, newOnes = a[i] & (~ones & ~twos) | ~a[i] & ones; newTwos = a[i] & ones | ~a[i] & twos;
// 扩展：若唯一一个不出现三次的数只出现两次?
// 扩展：若其他数出现 a 次，另一数出现 b 次 (b 不是 a 的位数)
solution: 模拟运算 a 进制 one, two, four, eight, 可用数组 BiDigit[lg a]，最后统计取模： 不可行 e.g. {2,2,2,3} 用两位, 则只剩下 {01}
// 扩展: 给定一个包含n个整数的数组，有一个整数x出现b次，一个整数y出现c次，其他所有的数均出现a次，其中b和c均不是a的倍数，找出x和y.
solution: 实现 mod a, 所得数为 b mod a次x，c mod a次y的累加.可证 该数肯定不为 0 ，找出不为零的一位，按此将所有数据分组再将问题简化。
// Extension: 若只有一数出现 k 次，其余出现 n 次 (k < n)? mod n

// 给一个string array, 除了一个string出现奇数次外，其他的所有string都出现了偶数次。返回出现奇数次的string
solution 1: 用 char *[] 对每个 char 做 XOR
solution 2: 可以用 set 来代替 XOR 的操作: initialize an empty set; scan every string,if its NOT in the set, put it into the set; otherwise, delete it from the set.  The remaining element is that string we want to find. 

// 数组长度为 N, 里面有一个数出现了大于N/2次，找出该数
solution: 相邻对消 num = A[0], cnt = 0; if (A[i-1] == A[i]) { cnt += 1; } elseif (cnt == 0) { num = A[i]; } else { cnt -= 1;}
// Extension: 若里面有一个数出现了大于 N/k 次呢？
solution: 每 K 个不同的数对消 (维持两个 vector<int> Num(k,0), Cnt(k,0); 及一个 int CntDiff, 用 hash table 更好 )

// Enable power x.
solution:
double pow(double x, int n) {// 2013.09.21 
    if(x == 1) return 1; // Error: x = 1, n = -23432434
    if(x == -1) return n%2 == 0?1:-1; // Error: x = -1, n = -23432434
    if(n<0){
      x = 1.0 / x; // x should be non-zero
      n = -n;
    }
    
    double result = 1;
    while(n){
        if(n & 1) result *= x;
        n >>= 1;
        x *= x;
    }
    return result;
}

// 用两个 unsigned 来实现一个元素在 0-9 之间的 queue 功能
