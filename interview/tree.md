Read records:
  * 2016.10.05 Not finished
  * 2013.11.12
  * 2013.10.23
  * 2013.10.04

Summaries:
  * 利用树型结构尽量用 recursive
  * 树型 recursive function 善于用"引用" 参数

Questions

* Construct Binary Tree from Preorder and Inorder Traversal
  // 需要 VC 找错误; 2013.05.01 // 在做完前题基础上还是有一错: 2013.05.01
	e.g. preorder = {7,10,4,3,1,2,8,11} inorder = {4,10,3,1,7,11,8,2} 求 tree
	==> solution: 用树型 recursive 算法，找出 root index. 其中可先 map inorder index
			==> Convert Vector to array: int * a = &v[0];
			==> while 里应比较 in[RootInorder] != pre[0]，但是 in[RootInorder] != pre[RootInorder]
			==> 如果按原题接口，large problem 有 memory not sufficient
			==> 比较边值问题 buildTreeArray(pre+1,in,RootInorder); buildTreeArray(pre+RootInorder+1,in+RootInorder+1,size-1-RootInorder);
		
	    TreeNode* buildTreeArray(const int*pre, const int*in, int size){
	        if(size == 0) return NULL;
	        TreeNode* root = new TreeNode(pre[0]);
	        int RootInorder = 0;
	        while(in[RootInorder] != pre[0] && RootInorder < size)++RootInorder;
	        root->left  = buildTreeArray(pre+1,in,RootInorder);
	        root->right = buildTreeArray(pre+RootInorder+1,in+RootInorder+1,size-1-RootInorder);
	        return root;
	    }
		    
	    TreeNode* buildTreeArray(const int*in, const int*pos, int size){
	        if(size == 0) return NULL;
	        TreeNode* root = new TreeNode(pos[size-1]);
	        int RootInorder = size-1;
	        while(in[RootInorder] != pos[size-1] && RootInorder >= 0)--RootInorder;
	        root->left  = buildTreeArray(in,pos,RootInorder);
	        root->right = buildTreeArray(in+RootInorder+1,pos+RootInorder,size-1-RootInorder);
	        return root;
	    }
	==> Extension 1: 可用 Inorder + Postorder Traversal; 
	==> Extension 2: 用 Preorder + Postorder 分不清左右子树; 
	==> Extension 3: Given preorder, inorder, and postorder traversal, how can you verify if these traversals are referring to the exact same binary tree? 


* 用 O(n) 算法来重建given array A[n] with unique elements 的 max-tree
  // max-tree, let m be the index of maximum element and A[0--m-1], A[m+1,--,n-1] 分别是 max-tree
	==> 从左到右建树，比较 TreeNode p 与 A[i], 若 A[i] > p->val 则 p 成 A[i] 的左子树，A[i] 代替 p 成 p 的 parent 的右子树; 否则 若 p 无右子树，则填入 a[i], 若有右子树 p = p->right.
	==> Extension: 根据 preorder 或者 postorder 重构树。 根据 inorder 不可能重构树

* Tree traverse without recursive
  * postorder: stack + flag(for unvisited branches, 也可不用 flag, 可用两个指针, check current node is the right child of the next node)
```
void PostOrder(TreeNode * h) {// 2013.11.08 简洁版
    vector<TreeNode*> q; 
    TreeNode* p = 0;
    while(h || !q.empty()){// 直接用 h != 0 or q is not empty() 控制循环
        while(h){ q.push_back(h); h = h->left;} // go to the left most node
        TreeNode* c = q.back();
        if(c->right && p != c->right){// right 是否存在，若存在, 是否访问过
            h = c->right;
        }else{
            visit(c);
            p= c;
            q.pop_back();
            h = 0;
        }
    }
    return result;
}
```
  * preorder: just stack
```
void PreOrder(node*t){// iterative approach using stack, unlike PostOrder, no need flag
	stack<node*> s(NULL); // last elment is NULL, control while
	while(t){
		if(t->right)s.push_back(t->right);
		if(t->left)s.push_back(t->left);
		visit(t);
		t = s.back(); s.pop_back();
	}
}
```
  * inorder: stack + flag; 可不用 flag (a) 直到最左，path 上所有点 push. 然后pop，若所 pop 的 node 有 right child, 对 right child 用到最左; (b) 可用 prev and curr 来判断是从子树返回与否。
```
void InOrder(Node *pRoot){// for ReverseInOrder, only switch the order of left child and right child
    Stack<Node *> s;
    while( pRoot || !s.Empty() ){
        while( pRoot ){// push most left leaf into stack, pRoot == NULL after this
            s.push_back(pRoot);
            pRoot = pRoot->pLeft;
        }
        pRoot = s.back(); s.pop_back();
        Visit(pRoot);
        pRoot = pRoot->pRight;
    }
}
```
  * Level Order: queue + CurCnt + NextCnt 

```
Binary Tree Level Order Traversal{// no error 2013.05.01
	==> solution: 用 队列 int CurrLevelCnt, NextLevelCnt;
	==> Extension: Binary Tree Zigzag Level Order Traversal // no error,无错通过 2013.05.01  但忘记清0， NextLvNum = 0; within while 2013.05.01
	--> solution: 用两个更替的 stack 即可
}
```
  * Zip-Zag Level Order: 利用 stack 先进后出的特性，用两个 stack 代表不同 level 



* 从左到右打印 BT 的叶子
	==> 递归
	==> 打印 BT 的边界 (root --> 最左叶子 --> 最右叶子 --> root): 用两个左右边界函数， 用 is_boundary 判断边界，若不是则不打印中间节点
}

* 给一个数组，判定是否可能是一个BST后序遍历得到的
// 用 [minVal, maxVal] 来界定子范围, 从而确定下个子树的开始 index，并更新 [minVal, maxVal]
	==> solution: 树型递归，parent 结点为分界点
```
	int scanPostOrder(int *A, int idx, int minVal, int maxVal){
          // idx 为 parent index, return 值为以 idx 为祖先的最前 index. 
	  if (idx < 0 || A[idx] < minVal || A[idx] > maxVal)return idx;
          //如果不能满足数值范围,或者已经扫描完毕，就返回到父节点
	  int val = A[idx];
	  idx = scanPostOrder(A, idx-1, val, maxVal);  // 右子树
	  return scanPostOrder(A, idx, minVal, val);   // 左子树
	}
	
	bool checkPostOrder(int *A, int n)
	{
	  return scanPostOrder(A, n-1, INT_MIN, INT_MAX) < 0;
	}
```

* Given a binary tree, find the lowest common ancestor of two given nodes in the tree
	==> Solution: recursive, bottom up. O(n)
```
	Node *LCA(Node *root, Node *p, Node *q) {// return root if both L R are not null, return L or R if only one of them is NOT null
	  if (!root) return 0;
	  if (root == p || root == q) return root;
	  Node *L = LCA(root->left, p, q);
	  Node *R = LCA(root->right, p, q);
	  if (L && R) return root;  // if p and q are on both sides
	  return L ? L : R;  // either one of p,q is on one side OR p,q is not in L&R subtrees
	}
```
	==> Extension: if nodes have pointers to parents, no need recursive. O(lg(n)), find the path of both nodes, estimate their hights, start from the same height, no need for extra space.

* convert a BST to a new Data Structure s.t.
	1. 所有 leaf node's 左 ptr 指向 its parent， 右 ptr 指向下一个 leaf node
	2. 所有 non leaf node's  左 ptr 指向 its parent，右 ptr == NULL
	3. return the head and print the new DS 
	==> solution: 从root 到一个 leaf 将所有 left 结点入 stack s,  然后对 s 中元素整理 left pointer. s1 Pop_back until empty or any element->right != 0;
```
	vector<Node*> s; // or head == h;
	StopInd = 0; // the index of the first node with non-empty right pointer in s
	Node P(0); Node * cur = &P;// leave's pointers
	while(!s.empty() || h){
		while(h){
			s.push_back(h);
			h = h->left;
		}
		cur->right = s.back(); cur = s.back();
		for(int i = s.size() - 1; i > StopInd; ++i) s[i]->left = s[i-1]; // update left pointer;
		s[0]->left = 0; // root's left pointer
		while(!s.empty() && s.back()->right == 0) s.pop_back(); // find the first element with non-empty right pointer
		if(s.size() > 0){
			StopInd = s.size() - 1;
			h = s.back()->right;
			s.back()->right = 0;
		}else{
			break;
		}
	}	
	return P.right;
```

* Sum Root to Leaf Numbers
  // 2013.09.22, every node in the tree has 0-9 digits, every leaf represent an int, e.g., 1-2-3 = 123, sum up all leaves.
```
	void sumNumbers_sub(TreeNode *root, int& sum, int path){
		if(!root) return;
		int p = path * 10 + root->val;
		if(!root->left && !root->right){
			sum += p;
			return;
		}
		sumNumbers_sub(root>left ,sum,p);
		sumNumbers_sub(root>right,sum,p);
	}
  int sumNumbers(TreeNode *root) {
  	int sum = 0;
  	sumNumbers_sub(root,sum,0);
  	return sum;
  }
```

* Path Sum
	==> Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all the values along the path equals the given sum.
```
    bool hasPathSum(TreeNode *root, int sum) {
        if(!root) return sum == 0;// 错误: 忘记判断是否为空
				if(hasPathSum(root->left,sum - root->val)) return true;
        if(hasPathSum(root->right,sum - root->val)) return true;
        return false;
    }
```
  ==> Extension: Given a binary tree and a sum, find all root-to-leaf paths where each path sum equals the given sum. // 2013.09.13 第二次无错通过
  	解决方法: 在函数形参里加 & result 和 & path

* Find the max sum path in a Binary tree, the start and end points are not necessary to be leaves or root.

	==> Solution: 2012.11.08: O(nlgn) or O(n) if allowing using O(lgn) space. Two recursive functions: f[i] and g[i], where g[i] is the maximum path from leaf to i; and f[i] is the maximum path within branch i: using stacks to calculate f[i], g[i] simultaneous
			f[i] = max{f[i->left],f[i->right],g[i->left]+g[i->right]+w[i]}, 
			g[i] = max{g[i->left],g[i->right]} + w[i], 
	
	妙在将 f[i] 完全放到子树 max 里update		
```
	int maxPath(TreeNode *root,int &max){// return the maximum top down sum; use max as the maximum: max=0;
	  if (!root)return 0;
	  int l = 0, r = 0; 
	  if(root->left) l = maxPath(root->left,max);// g[i->left]; max is f[i]
	  if(root->right)r = maxPath(root->right,max);// g[i->left]; max is f[i]
	  int MaxWithRoot = l+r+root->val;
	  max = MaxWithRoot>max?max:MaxWithRoot; // update f[i]
	  MaxWithRoot = (l>r?l:r)+root->val;// g[i]
	  return MaxWithRoot>0?MaxWithRoot:0;
	}
```
	==> Extension: 找 all paths which sum up to a given value. Note that it can be any path in the tree - it does not have to start at the root.	
	==> Solution: Time O(nlgn) space O(lgn) CareerCup 上的解释是从 root 出发向下走的 sum . 其实 若如题所示，考虑到左右子树，很难做到。iterative, using stack. Hard problem, the key is how to find a path through a node with the sum == given value. One need to test every possible combinations of left branch and right branch

* Set all in order-successor pointers of the given binary search tree// use post-order to set 
You are given a binary search tree T of finite (means can fit in memory) size n in which each node N contains
- integer data element
- pointer to left child
- pointer to right child
- pointer to in order successor (which is set null for each node)

* Set all in order successor pointers of the given binary search tree

==> using stack, iterative. [For problems need previous pointers, iterative is much better than recursive]

==> Recursive Version: 
```
Node* SetInOrderSuccessor(Node*T, Node* Pre){// 需要 前面结点指针 return the last in-order element
	if(!T) return 0;	
	if(T->left)
		SetInOrderSuccessor(T->left,Pre)->successor = T;
	else
		if(Pre) Pre->successor = T;

	if(T->right) T = SetInOrderSuccessor(T->right,T);
	return T;
}
```

* Serialization/Deserialization of a Binary Tree
	==> solution: use preorder + '#' 代表 null. 用 vector<Node*> q; 和一个 CurrentIndex 
	==> Deserialize: 用 queue

* find the kth element /the last kth element in BST // 传 k 的引用
==> Solution: O(k+lgn), with in-order traverse(min) or reverse-in-order(max) 差别在于左右子树的次序
```
	node* InOrder( node*t, int& k){// t != NULL, k > 0
		if(k <=0 || t == 0) return 0;
		node * Result = InOrder(t->left, k);
		if(Result) return Result;
		if(k-- == 1) return t;
		return InOrder(t->right, k);
	}
```

* Find the closest k-th elements in a BST for a given x// 2012.11.08
	==> FindkthClosestLarger(), FindkthClosestSmaller() 分别找出比 x 大的最小的 k 个 element 用 O(k+lgn), 再 merge O(k). 

* 找两个二叉树最大的相同子树 // hash value
	==> solution: hash(node) to double: e.g. ln5 * val + ln2 * hash(left) + ln3 * hash(right);
	==> Extension:	给一堆 Binary tree，合并相同子树 // element of programming interview Q 12.4 
	==> 关键是 hash function: h(full) = 1, h(n) = 3*val + 5 * h(n->left) + 7 * h(n->right); 
		// 觉得有问题 应该是 素数的累乘, 取 lg 后应为 h(n) = ln 3* val + ln 5 * h(l) + ln 7 * h(r);
```
Same Tree{// no error pass
	  bool isSameTree(TreeNode*t1,TreeNode*t2){
        if(t1==NULL || t2 == NULL) return t1 == t2;
        return t1->val != t2->val && isSameTree(t1->left,t2->left) && isSameTree(t1->right,t2->right);
    }
}
```

* Symmetric Tree
		==> 错误
			* 递归时应用 t1->left 与 t2->right 交错比较，而不是 left v.s. right.
			* if(t1->val != t1->val) return false; 里有 typo, 第二个 t1->val 应为 t2->val

```
    bool isSymmetricTree(TreeNode *t1, TreeNode* t2) {// 
    	if(t1==NULL || t2 == NULL) return t1 == t2;
    	return t1->val != t2->val && isSymmetricTree(t1->left,t2->right) && isSameTree(t1->right,t2->left);
    }
```

* Unqiue BST// Num of unique BST with n distinct numbers
```
    int numTrees(int n) {// 2013.09.23 DP convolution: f_k = f_{k-1} + f_{k-2}  * f_1+ .... +f_1 * f{k-2} + f_{k-1}
        if(n < 3) return n;
        int* r = new int[n+1]; 
        r[0] = 1;
        r[1] = 1;
        for(int i = 2; i <= n; ++i){
        	r[i] = 0; //Error: forget to initialize r[i]
            for(int k = 0; k < i/2; ++k){
                r[i] += r[k]*r[i-1-k];
            }
            r[i] += r[i];
            if(i%2) r[i] += r[i/2] * r[i/2];
        }
        int result = r[n];
        delete[] r;
        return result;
    }
```

* Unqiue BST II// Generate all BST, 用递归，但要注意不要重复劳动
```
		TreeNode * CopyAddTrees(TreeNode * OldTree, int AddNum){
			if(OldTree == 0) return 0;
			TreeNode * temp = new TreeNode(OldTree->val+AddNum);
			temp->left = CopyAddTrees(OldTree->left,AddNum);
			temp->right = CopyAddTrees(OldTree->right,AddNum);
			return temp;
		}
		
		void generateTrees_Sub(int NodeNum, vector< vector<TreeNode *> > & Result){// Result has NodeNum elements from 0 -- NodeNum-1
			vector<TreeNode *> NewTreeSet;
			if(NodeNum == 1){// NodeNum == 1 is special, cannot converge with others
				NewTreeSet.push_back(new TreeNode(1));
				Result.push_back(NewTreeSet);
				return;
			}
			
			for(int i = 0; i < Result.size(); ++i){// i is the position of the node NodeNum
				vector<TreeNode *> RightChildren;
				RightChildren.clear();
				for(int k = 0; k < Result[NodeNum-1-i].size(); ++k){
					RightChildren.push_back(CopyAddTrees(Result[NodeNum-1-i][k],i));
				}
				for(int j = 0; j < i; ++j){
					for(int k = 0; k < RightChildren.size(); ++k){
							TreeNode * root = new TreeNode(i);
							root->left = Result[i-1][j];
							root->right = RightChildren[k];
							NewTreeSet.push_back(root);
					}
				}
			}
			Result.push_back(NewTreeSet);
		}
		
    vector<TreeNode *> generateTrees(int n) {// 2013.09.26 DP Generate all BST
    	vector<TreeNode *> temp(1,0);
    	if(n <= 0) return temp;
    	vector< vector<TreeNode *> > Result;
    	Result.push_back(temp);
      generateTrees_Sub(n, Result);    	
    	return Result[n];
    }
```

* Maximum Depth of Binary Tree// 有一错: int leftDep = rightDep = 0; 有 compile 错误 2013.05.01
```
    int maxDepth(TreeNode *root) {
        if(!root) return 0;
        return max(maxDepth(root->left),maxDepth(root->right)) + 1;
    }
```

* Minimum Depth of Binary Tree
//  有错，对于仅有一子的node depth 只计到该 node. 2013.05.02
```
    int minDepth(TreeNode *root) {
        if(!root) return 0;
        return min(minDepth(root->left),minDepth(root->right)) + 1;
    }
```

* Balanced Binary Tree//  2013.05.01 左右子树最多相差 1
```

    bool isBalanced(TreeNode *root) {
        return TreeMaxDepth(root);
    }
    bool TreeMaxDepth(TreeNode *root, int& MaxDepth){
        if(!root){
        	MaxDepth = 0;
        	return true;
        }
        int l,r;
        if(!TreeMaxDepth(root->left,l)) return false;
        if(!TreeMaxDepth(root->right,r)) return false;
        MaxDepth = max(l,r)+1;
        return l-r <= 1 && r-l <= 1;
    }
```

* Convert Sorted Array to Binary Search Tree// 出错: mid 没有定义 2013.05.01
```
    TreeNode *sortedArrayToBST_sub(const int* n, int size) {
        if(size == 0) return NULL;
        int mid = size >> 2;
        TreeNode* root = new TreeNode(n[mid]);
        root->left = sortedArrayToBST_sub(n,mid);
        root->right = sortedArrayToBST_sub(n+mid+1,size-mid-1);
        return root;
    }
```

* Convert Sorted List to Binary Search Tree// 与 array 可以知道 size 不一样， 需要找到size. 两个 pointers, step 分别是1和2. 2013.05.01, 2013.09.22
```
    TreeNode *sortedListToBST(ListNode *head) {
        if(head == 0) return NULL;
        ListNode p(0); 
        p.next = head;
        head = &p;
        ListNode *f = head;
        while(f->next){// 出错: 必须将 prep1 = prep1->next; 放在 最后
        	f = f->next;
        	if(f->next ==  0) break;
        	head = head->next;        	
        	f = f->next; 
        }
        TreeNode* root = new TreeNode(head->next->val);
        root->right = sortedListToBST(head->next->next);
        if(&p != head){
	        head->next    = 0;
	        root->left  = sortedListToBST(p.next);
	      }
        return root;
    }
```

* Recover Binary Search Tree// 2013.09.22 用上下界指针判断, 其中两结点出错
	==> solution: 用 inorder travel 保持 pre 找出其两个指针位置，皆满足 cur < pre， 再转换则可
	// http://fisherlei.blogspot.com/2012/12/leetcode-recover-binary-search-tree.html

* Flatten Binary Tree to Linked List// 2013.05.02 VC 无错，但是 Leetcode 上有错
```
    void flatten(TreeNode *root) {
        lastNodePt(root);
    }
    TreeNode * lastNodePt(TreeNode*root){// 2013.10.24 修改
			if(!root) return 0;
			TreeNode * tempP = lastNodePt(root->left);
			tempP = tempP?root:tempP;
			tempP->right = root->right;
			return lastNodePt(root->right);
    }
```

* Populating Next Right Pointers in Each Node{// 2013.09.22 无错通过
	==> 思想：用递归，然后从左右子树开始，右子树走最左，左子树走最右，把同一层相连即可
  ==> Solution 2: 用 stack 从 root 走最左边界，每一结点记录层数
}
    
* Validate Binary Search Tree
// 出错: (1) 无论左右子树都需要最大值与最小值 (2) 引用不能有默认参数 TreeNode*&m=NULL 出错
```
    bool isValidBST(TreeNode *root) {
        return MaxMinBST(root,INT_MIN-1ll,INT_MAX+1ll));
    }
    bool MaxMinBST(TreeNode*root, long long m, long long M){// return NULL if t is not BST        
        if(!root) return true;
        return MaxMinBST(root->left,m,root->val) && MaxMinBST(root->right,root->val,M) && (root->val > m && root->val < M);
    }
```

* Given n-strings, 找 string chain (即前 string 最后一个字母 == 后 string 第一个字母)
	 e.g. 3 strings: "sdfg","dfgs","ghjhk"; chain is "dfgsdfghjhk"
	 ==> solution: full path pass all nodes, topological sort, 
	 	1. vector<int> a; vector< vector<int> > FirstLetter(26,a),LastLetter(26,a);


* Give a list of events in the following structure. Set the conflict flag to true if the event conflicts with any other event in the list
```
	class Event
	{
	    int start;
	    int end;
	    bool conflict;
	}
```
	==> solution: interval tree construct O(nlgn) inquire O(lgn+m)

