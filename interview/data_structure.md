// ================================================================= //
阅读记录: 2013.10.25
// ================================================================= //

Wiki{
http://en.wikipedia.org/wiki/Suffix_tree
http://en.wikipedia.org/wiki/Suffix_array
http://en.wikipedia.org/wiki/Radix_tree
}

Trie Tree {
==> Trie Tree (or prefix tree) "edge == one letter", while Suffix Tree "edge == a string", which is compressed from a single child path in trie since all internal nodes in Suffix requires at least two children and thus Suffix at most has 2n nodes.
==> Trie, all words come from a dictionary. While Suffix tree, all words come from suffice of a string. (or suffice of a set of strings for General Suffix tree). 
==> Relation: Suffix \subset Radix (Patricia) \subset Trie. Radix is a compact trie. For each node in Radix tree, just needs two pointers to present: the first of the substring and the last of the substring. Radix is not limited to Suffice of a string.
}

Trie Tree {
Construction and Implementation

http://www.cs.helsinki.fi/u/ukkonen/SuffixT1withFigs.pdf

==> Building a Trie O(n^2); Suffix O(n) online (Ukkonen,1995) for constant size alphabet set(since no more than 2n nodes). or O(nlgn) for infinte alphabet set.

==> An important choice when making a suffix tree implementation is the parent-child relationships between nodes. The most common is using linked lists called sibling lists. Let \sigma be the size of the alphabet. Then you have the following costs:
			                                Lookup 	      Insertion 	Traversal
		Sibling lists / unsorted arrays: O(\sigma) 	    \Theta(1) 	   \Theta(1)
		Hash maps 	                   :\Theta(1)  	    \Theta(1) 	   O(\sigma)
		Balanced search tree 	         : O(\log \sigma) O(\log \sigma) O(1)
		Sorted arrays 	               : O(\log \sigma) O(\sigma) 	   O(1)
		Hash maps + sibling lists 	   : O(1) 	        O(1) 	         O(1)

node* SuffixTreeBuild(s){// Ukkonen algorithm O(n), s should be end with $.
	if(s.empty())return;
	int i  = 0; // input substring index
	int si; // unbranched suffix starting index

	node * root = new node; // node initially has len = -1; meaning to the end;
	node * pre  = root; // prenode to make suffix link
	node * act  = NULL; // active node
	
	while(i < s.size()){
		if(root[s[i]] == NULL){// new starting letter
			if(act){// if has active node
				while(si < i){// break nodes
					int len = i - si; // total length of unbranced suffix
					while(act.len > 0 && act.len < len){ // find the proper node
						len -= act.len;
						act = act->child[s[si+act.len]];
					}
					node * cur = new node; // 
					cur.start = act.start;
					cur.len = len;
					act.start = cur.start + cur.len;
					cur->child[s[act.start]] = act;
					pre->SuffixLink = cur;
					pre = cur;
					++si;
				}
				act = NULL;
			}else{ // if node active node
				root->child[s[i]] = new node;
				root->child[s[i]].start = i;
				pre->SuffixLink = root->child[s[i]];
				pre = root->child[s[i]];
				si = i+1;
			}
		}else{// new starting letter
			act = root->child[s[i]];
		}
		++i;
	}
	return root;
}

}

Applications{
==> Finding the longest repeated substring O(n): Suffix find the deepest node that has at least two suffice. 
==> Finding the longest common substring O(n): build Suffix with all suffice or both words. find the deepest node that has common suffice
==> Finding the longest palindrome in a string O(n): build Suffix with all prefice and suffice. find the deepest node that has both prefix and suffix
==> Check a length = m substring O(min(n,m))
==> Count a substring's repetition O(n)
==> Find the most frequently occurring substrings of a minimum length in \Theta(n) time.
==> Find the longest substrings common to at least k strings in D for k=2,\dots,K in \Theta(n) time
==> Find the shortest substrings occurring only once in \Theta(n) time.
==> ? Find the shortest strings from \Sigma that do not occur in D, in O(n + z) time, if there are z such strings.: build Suffix Tree for \Sigma, 
==> ? Find, for each i, the shortest substrings of S_i not occurring elsewhere in D in \Theta(n) time.
==> Find the first occurrence of the patterns P_1,\dots,P_q of total length m as substrings in O(m): build a suffix tree for P_i, then compare two suffix trees.
==> Find all z occurrences of the patterns P_1,\dots,P_q of total length m as substrings in O(m + z): 

}

Suffix Array: sort all the suffix. {
http://en.wikipedia.org/wiki/Suffix_array
}


Search trees for non-string data(spacial data){

==> B(X)-tree ==> R(X)-tree (rectangular tree), M-tree (ball )

}

k-d tree{
	http://en.wikipedia.org/wiki/K-d_tree
	多维 binary tree, 即每个结点分割所在维度不一样
	有多种构造方法，其中 a balanced k-d tree, 
	1. cycle dimension x-y-z-x-y-z
	2. median splitting value
}

// ================================================================= //
Binary trees 	

    Binary search tree (BST)
    Cartesian tree
    MVP Tree
    Top tree
    T-tree
    Left-child right-sibling binary tree

Self-balancing binary search trees 	

    AA tree
    AVL tree
    LLRB tree
    Red–black tree
    Scapegoat tree
    Splay tree
    Treap

B-trees 	

    B+ tree
    B*-tree
    Bx-tree
    UB-tree
    2–3 tree
    2–3–4 tree
    (a,b)-tree
    Dancing tree
    HTree

Tries 	

    Suffix tree
    Radix tree
    Hash tree
    Ternary search tree
    X-fast trie
    Y-fast trie

Binary space partitioning (BSP) trees 	

    Quadtree (用于 2D)
    Octree (用于 3D)
    k-d tree
    Implicit k-d tree
    VP tree

Non-binary trees 	

    Exponential tree
    Fusion tree
    Interval tree
    PQ tree
    Range tree
    SPQR tree
    Van Emde Boas tree

Spatial data partitioning trees 	

    R-tree ( rectangles 二维 interval tree)
    R+ tree
    R* tree
    X-tree (based on R-tree but emphasizes prevention of overlap)
    M-tree ( metric-tree relies on triangle condition
    Segment tree
    Hilbert R-tree
    Priority R-tree

Other trees 	
    Heap
    Hash calendar
    Merkle tree
    Finger tree
    Order statistic tree
    Metric tree
    Cover tree
    BK-tree
    Doubly chained tree
    iDistance
    Link/cut tree
    Fenwick tree
    Log-structured merge-tree
// ================================================================= //


Bloom Filter+Bitmap{// 用于海量数据的查找，允许 false positive error
	K 个独立的hash function overlap 在一个 bitmap m 位上判断，并不保证查找的结果是100%正确的。同时也不支持删除一个已经插入的关键字。设数据数为 n, 则 bitmap 上某位为 0 的概率是 (1-1/m)^{kn}:= p, 错误率为 (1-p)^k := f. 可 optimize f over k,m while n fixed. 
	
	Counting bloom filter（CBF）将位数组中的每一位扩展为一个counter，从而支持了元素的删除操作。Spectral Bloom Filter（SBF）将其与集合元素的出现次数关联。SBF采用counter中的最小值来(近似)表示元素的出现频率。 
}

从头到尾彻底解析Hash表算法{
	http://blog.csdn.net/v_JULY_v/article/details/6256463

 MPQ使用文件名哈希表来跟踪内部的所有文件。但是这个表的格式与正常的哈希表有一些不同。首先，它没有使用哈希作为下标，把实际的文件名存储在表中用于验证，实际上它根本就没有存储文件名。而是使用了3种不同的哈希：一个用于哈希表的下标，两个用于验证。这两个验证哈希替代了实际文件名。
}
// ================================================================= //
multi-thread
http://www.zybuluo.com/smilence/note/540

字符串的近似匹配
http://haoyuheng680.blog.163.com/blog/static/5001601020101015101727392/


Rolling hash{// http://en.wikipedia.org/wiki/Rolling_hash
}

Least Recently Used (LRU) cache: supporting get() and set(){
	==> map + double linkedlist 
	 * Node head, tail; 来管理双向 linkedlist
	 * 双向 linkedlist 对于调整 node 的次序很有用
}