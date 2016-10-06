Read records:
  * 2016.10.05 Not finished
  * 2013.11.12
  * 2013.10.23
  * 2013.10.04

Summaries:
  * �������ͽṹ������ recursive
  * ���� recursive function ������"����" ����

Questions

* Construct Binary Tree from Preorder and Inorder Traversal
  // ��Ҫ VC �Ҵ���; 2013.05.01 // ������ǰ������ϻ�����һ��: 2013.05.01
	e.g. preorder = {7,10,4,3,1,2,8,11} inorder = {4,10,3,1,7,11,8,2} �� tree
	==> solution: ������ recursive �㷨���ҳ� root index. ���п��� map inorder index
			==> Convert Vector to array: int * a = &v[0];
			==> while ��Ӧ�Ƚ� in[RootInorder] != pre[0]������ in[RootInorder] != pre[RootInorder]
			==> �����ԭ��ӿڣ�large problem �� memory not sufficient
			==> �Ƚϱ�ֵ���� buildTreeArray(pre+1,in,RootInorder); buildTreeArray(pre+RootInorder+1,in+RootInorder+1,size-1-RootInorder);
		
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
	==> Extension 1: ���� Inorder + Postorder Traversal; 
	==> Extension 2: �� Preorder + Postorder �ֲ�����������; 
	==> Extension 3: Given preorder, inorder, and postorder traversal, how can you verify if these traversals are referring to the exact same binary tree? 


* �� O(n) �㷨���ؽ�given array A[n] with unique elements �� max-tree
  // max-tree, let m be the index of maximum element and A[0--m-1], A[m+1,--,n-1] �ֱ��� max-tree
	==> �����ҽ������Ƚ� TreeNode p �� A[i], �� A[i] > p->val �� p �� A[i] ����������A[i] ���� p �� p �� parent ��������; ���� �� p ���������������� a[i], ���������� p = p->right.
	==> Extension: ���� preorder ���� postorder �ع����� ���� inorder �������ع���

* Tree traverse without recursive
  * postorder: stack + flag(for unvisited branches, Ҳ�ɲ��� flag, ��������ָ��, check current node is the right child of the next node)
```
void PostOrder(TreeNode * h) {// 2013.11.08 ����
    vector<TreeNode*> q; 
    TreeNode* p = 0;
    while(h || !q.empty()){// ֱ���� h != 0 or q is not empty() ����ѭ��
        while(h){ q.push_back(h); h = h->left;} // go to the left most node
        TreeNode* c = q.back();
        if(c->right && p != c->right){// right �Ƿ���ڣ�������, �Ƿ���ʹ�
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
  * inorder: stack + flag; �ɲ��� flag (a) ֱ������path �����е� push. Ȼ��pop������ pop �� node �� right child, �� right child �õ�����; (b) ���� prev and curr ���ж��Ǵ������������
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
	==> solution: �� ���� int CurrLevelCnt, NextLevelCnt;
	==> Extension: Binary Tree Zigzag Level Order Traversal // no error,�޴�ͨ�� 2013.05.01  ��������0�� NextLvNum = 0; within while 2013.05.01
	--> solution: ����������� stack ����
}
```
  * Zip-Zag Level Order: ���� stack �Ƚ���������ԣ������� stack ����ͬ level 



* �����Ҵ�ӡ BT ��Ҷ��
	==> �ݹ�
	==> ��ӡ BT �ı߽� (root --> ����Ҷ�� --> ����Ҷ�� --> root): ���������ұ߽纯���� �� is_boundary �жϱ߽磬�������򲻴�ӡ�м�ڵ�
}

* ��һ�����飬�ж��Ƿ������һ��BST��������õ���
// �� [minVal, maxVal] ���綨�ӷ�Χ, �Ӷ�ȷ���¸������Ŀ�ʼ index�������� [minVal, maxVal]
	==> solution: ���͵ݹ飬parent ���Ϊ�ֽ��
```
	int scanPostOrder(int *A, int idx, int minVal, int maxVal){
          // idx Ϊ parent index, return ֵΪ�� idx Ϊ���ȵ���ǰ index. 
	  if (idx < 0 || A[idx] < minVal || A[idx] > maxVal)return idx;
          //�������������ֵ��Χ,�����Ѿ�ɨ����ϣ��ͷ��ص����ڵ�
	  int val = A[idx];
	  idx = scanPostOrder(A, idx-1, val, maxVal);  // ������
	  return scanPostOrder(A, idx, minVal, val);   // ������
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
	1. ���� leaf node's �� ptr ָ�� its parent�� �� ptr ָ����һ�� leaf node
	2. ���� non leaf node's  �� ptr ָ�� its parent���� ptr == NULL
	3. return the head and print the new DS 
	==> solution: ��root ��һ�� leaf ������ left ����� stack s,  Ȼ��� s ��Ԫ������ left pointer. s1 Pop_back until empty or any element->right != 0;
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
        if(!root) return sum == 0;// ����: �����ж��Ƿ�Ϊ��
				if(hasPathSum(root->left,sum - root->val)) return true;
        if(hasPathSum(root->right,sum - root->val)) return true;
        return false;
    }
```
  ==> Extension: Given a binary tree and a sum, find all root-to-leaf paths where each path sum equals the given sum. // 2013.09.13 �ڶ����޴�ͨ��
  	�������: �ں����β���� & result �� & path

* Find the max sum path in a Binary tree, the start and end points are not necessary to be leaves or root.

	==> Solution: 2012.11.08: O(nlgn) or O(n) if allowing using O(lgn) space. Two recursive functions: f[i] and g[i], where g[i] is the maximum path from leaf to i; and f[i] is the maximum path within branch i: using stacks to calculate f[i], g[i] simultaneous
			f[i] = max{f[i->left],f[i->right],g[i->left]+g[i->right]+w[i]}, 
			g[i] = max{g[i->left],g[i->right]} + w[i], 
	
	���ڽ� f[i] ��ȫ�ŵ����� max ��update		
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
	==> Extension: �� all paths which sum up to a given value. Note that it can be any path in the tree - it does not have to start at the root.	
	==> Solution: Time O(nlgn) space O(lgn) CareerCup �ϵĽ����Ǵ� root ���������ߵ� sum . ��ʵ ��������ʾ�����ǵ���������������������iterative, using stack. Hard problem, the key is how to find a path through a node with the sum == given value. One need to test every possible combinations of left branch and right branch

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
Node* SetInOrderSuccessor(Node*T, Node* Pre){// ��Ҫ ǰ����ָ�� return the last in-order element
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
	==> solution: use preorder + '#' ���� null. �� vector<Node*> q; ��һ�� CurrentIndex 
	==> Deserialize: �� queue

* find the kth element /the last kth element in BST // �� k ������
==> Solution: O(k+lgn), with in-order traverse(min) or reverse-in-order(max) ����������������Ĵ���
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
	==> FindkthClosestLarger(), FindkthClosestSmaller() �ֱ��ҳ��� x �����С�� k �� element �� O(k+lgn), �� merge O(k). 

* ������������������ͬ���� // hash value
	==> solution: hash(node) to double: e.g. ln5 * val + ln2 * hash(left) + ln3 * hash(right);
	==> Extension:	��һ�� Binary tree���ϲ���ͬ���� // element of programming interview Q 12.4 
	==> �ؼ��� hash function: h(full) = 1, h(n) = 3*val + 5 * h(n->left) + 7 * h(n->right); 
		// ���������� Ӧ���� �������۳�, ȡ lg ��ӦΪ h(n) = ln 3* val + ln 5 * h(l) + ln 7 * h(r);
```
Same Tree{// no error pass
	  bool isSameTree(TreeNode*t1,TreeNode*t2){
        if(t1==NULL || t2 == NULL) return t1 == t2;
        return t1->val != t2->val && isSameTree(t1->left,t2->left) && isSameTree(t1->right,t2->right);
    }
}
```

* Symmetric Tree
		==> ����
			* �ݹ�ʱӦ�� t1->left �� t2->right ����Ƚϣ������� left v.s. right.
			* if(t1->val != t1->val) return false; ���� typo, �ڶ��� t1->val ӦΪ t2->val

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

* Unqiue BST II// Generate all BST, �õݹ飬��Ҫע�ⲻҪ�ظ��Ͷ�
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

* Maximum Depth of Binary Tree// ��һ��: int leftDep = rightDep = 0; �� compile ���� 2013.05.01
```
    int maxDepth(TreeNode *root) {
        if(!root) return 0;
        return max(maxDepth(root->left),maxDepth(root->right)) + 1;
    }
```

* Minimum Depth of Binary Tree
//  �д����ڽ���һ�ӵ�node depth ֻ�Ƶ��� node. 2013.05.02
```
    int minDepth(TreeNode *root) {
        if(!root) return 0;
        return min(minDepth(root->left),minDepth(root->right)) + 1;
    }
```

* Balanced Binary Tree//  2013.05.01 �������������� 1
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

* Convert Sorted Array to Binary Search Tree// ����: mid û�ж��� 2013.05.01
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

* Convert Sorted List to Binary Search Tree// �� array ����֪�� size ��һ���� ��Ҫ�ҵ�size. ���� pointers, step �ֱ���1��2. 2013.05.01, 2013.09.22
```
    TreeNode *sortedListToBST(ListNode *head) {
        if(head == 0) return NULL;
        ListNode p(0); 
        p.next = head;
        head = &p;
        ListNode *f = head;
        while(f->next){// ����: ���뽫 prep1 = prep1->next; ���� ���
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

* Recover Binary Search Tree// 2013.09.22 �����½�ָ���ж�, ������������
	==> solution: �� inorder travel ���� pre �ҳ�������ָ��λ�ã������� cur < pre�� ��ת�����
	// http://fisherlei.blogspot.com/2012/12/leetcode-recover-binary-search-tree.html

* Flatten Binary Tree to Linked List// 2013.05.02 VC �޴����� Leetcode ���д�
```
    void flatten(TreeNode *root) {
        lastNodePt(root);
    }
    TreeNode * lastNodePt(TreeNode*root){// 2013.10.24 �޸�
			if(!root) return 0;
			TreeNode * tempP = lastNodePt(root->left);
			tempP = tempP?root:tempP;
			tempP->right = root->right;
			return lastNodePt(root->right);
    }
```

* Populating Next Right Pointers in Each Node{// 2013.09.22 �޴�ͨ��
	==> ˼�룺�õݹ飬Ȼ�������������ʼ�������������������������ң���ͬһ����������
  ==> Solution 2: �� stack �� root ������߽磬ÿһ����¼����
}
    
* Validate Binary Search Tree
// ����: (1) ����������������Ҫ���ֵ����Сֵ (2) ���ò�����Ĭ�ϲ��� TreeNode*&m=NULL ����
```
    bool isValidBST(TreeNode *root) {
        return MaxMinBST(root,INT_MIN-1ll,INT_MAX+1ll));
    }
    bool MaxMinBST(TreeNode*root, long long m, long long M){// return NULL if t is not BST        
        if(!root) return true;
        return MaxMinBST(root->left,m,root->val) && MaxMinBST(root->right,root->val,M) && (root->val > m && root->val < M);
    }
```

* Given n-strings, �� string chain (��ǰ string ���һ����ĸ == �� string ��һ����ĸ)
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

