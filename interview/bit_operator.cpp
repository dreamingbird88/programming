// ====================================================================== //
// Read Record 2013.10.23
// ====================================================================== //

// Swap odd bits with even bits in integer. E.g. swap bit 0 and bit 1, bit 2
// and bit 3: 10001010 --> 01000101
solution: �� 0xaaaaaa 0x555555 �ҳ���żλ ��λ ���� | �ಢ

// ���λ 1 ����
solution: i & (i - 1)
  
// ȡ���λ 1
solution: i & !(i-1);  

// Build a mask, s.t. [i,j] bits == 1, while other bits == 0.
// If ~mask, then [i, j] == 0, while other bits == 1.
solution: (1) [j-i,0] ȫΪ 1: (1<<(j-i+1) - 1);
          (2) ���� i λ: Mask = (1 << (j-i+1) - 1) << i

// Check if 'n' is a power of 2. I.e. n = 2, 4, 8, ...
solution: �����Ƿ�ֻ��һ�� 1 	((n & (n-1)) == 0)
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
solution: ������λ 5 ��������ʾ 1-7.
int num = 5*(rand5()-1) + rand5() - 1; if (num < 21) return num % 7;
// can use multiple bits of "5" to represent multiple bits of "7"


// In an int array, only one number appear once, others are twice. Find that
// number. Ҫ�����һ�����
solution: �� bit XOR (^) ȥ�����������ε���
// Extension: ����������������?
solution: ��һ���� count ÿλ�ϵ� bit, �� mod 3:  ones ��ʾ����һ�ε�, twos ��ʾ�������ε�, newOnes = a[i] & (~ones & ~twos) | ~a[i] & ones; newTwos = a[i] & ones | ~a[i] & twos;
// ��չ����Ψһһ�����������ε���ֻ��������?
// ��չ�������������� a �Σ���һ������ b �� (b ���� a ��λ��)
solution: ģ������ a ���� one, two, four, eight, �������� BiDigit[lg a]�����ͳ��ȡģ�� ������ e.g. {2,2,2,3} ����λ, ��ֻʣ�� {01}
// ��չ: ����һ������n�����������飬��һ������x����b�Σ�һ������y����c�Σ��������е���������a�Σ�����b��c������a�ı������ҳ�x��y.
solution: ʵ�� mod a, ������Ϊ b mod a��x��c mod a��y���ۼ�.��֤ �����϶���Ϊ 0 ���ҳ���Ϊ���һλ�����˽��������ݷ����ٽ�����򻯡�
// Extension: ��ֻ��һ������ k �Σ�������� n �� (k < n)? mod n

// ��һ��string array, ����һ��string�����������⣬����������string��������ż���Ρ����س��������ε�string
solution 1: �� char *[] ��ÿ�� char �� XOR
solution 2: ������ set ������ XOR �Ĳ���: initialize an empty set; scan every string,if its NOT in the set, put it into the set; otherwise, delete it from the set.  The remaining element is that string we want to find. 

// ���鳤��Ϊ N, ������һ���������˴���N/2�Σ��ҳ�����
solution: ���ڶ��� num = A[0], cnt = 0; if (A[i-1] == A[i]) { cnt += 1; } elseif (cnt == 0) { num = A[i]; } else { cnt -= 1;}
// Extension: ��������һ���������˴��� N/k ���أ�
solution: ÿ K ����ͬ�������� (ά������ vector<int> Num(k,0), Cnt(k,0); ��һ�� int CntDiff, �� hash table ���� )

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

// ������ unsigned ��ʵ��һ��Ԫ���� 0-9 ֮��� queue ����
