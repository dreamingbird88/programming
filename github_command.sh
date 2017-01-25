# References:
#   https://www.atlassian.com/git/tutorials/using-branches
#   https://git-scm.com/docs
#
# 2015.06.25 Learn to use github.
sudo apt-get install git

# Check config.
git config --list
sudo git config --system core.editor vim
git config --global user.email jeffnju@gmail.com
git config --global user.name dreamingbird88

git clone https://github.com/dreamingbird88/kaggle
git remote add name https://github.com/dreamingbird88/kaggle/
git remote add program https://github.com/dreamingbird88/programming/


git add programming/github_command.jeff
#git status
git commit -m "Upload github_command"
git push -u origin master

# 1. First create a repository with browser.
# 2. create a new repository on the command line
echo "# for_qi" >> README.md
git init
git add README.md
git commit -m "first commit"
git clone https://github.com/dreamingbird88/for_qi
git remote add for_qi https://github.com/dreamingbird88/for_qi

git remote set-url programming https://github.com/dreamingbird88/programming

git push -u program master

git commit -a -m "python script combining passport photos for print"

git add array_linklist_string.md  graph.md     
git add math.html

git fetch for_qi master

git add python/qi_skull.py
git commit -m "Add timer for math.html"
git push
git push -u programming master

is_new=true
project="ML4T_2017Spring/mc1p2_optimize_portfolio/optimization.py"
project="ML4T_2017Spring/mc1p1_assess_portfolio/analysis.py"
git_dest=$githubdir/mscs/${project}
mkdir -p  ${git_dest%/*}

qi_source=$qidir/temp/${project} 
if [ is_new ]; then
  qi_source=$qidir/${project} 
fi

cp -p ${qi_source} ${git_dest%/*}

cd ${githubdir}
git add mscs/${project}
git commit -m "Upload the original files for mscs homework."
git push

# user
dreamingbird88
# pw
gdlj;jeff1980

ls

# Delete a repository
# go to https://github.com/user_name/repository_name/setting
# dangerous zone --> delete ...
# local: rm -f -r dir_name

# # or push an existing repository from the command line
# git remote add origin https://github.com/dreamingbird88/finance.git
# git push -u origin master

# g4 sync
git pull https://github.com/dreamingbird88/programming/

# Create a new repository
# Can NOT create a new repository AND update it by "git ..."
# git init Finance
git remote add finance https://github.com/dreamingbird88/finance/
git clone finance
# git clone https://github.com/dreamingbird88/for_qi

# g4 change
git config --global user.email jeffnju@gmail.com
git config --global user.name dreamingbird88
git commit

# check the commit (change list)
git status

git push

git commit --amend
vim .gitignore
git log
git checkout
git add to_be_deleted.txt 
git checkout
git commit -m "First message"
git add another_test.txt 
git checkout
git status
git commit -m "second message"
git log --oneline
git reset HEAD~2
git add .gitignore 
vim .gitignore
git add .gitignore 
git commit -m "add .gitignore"
git log
git clean -df
git checkout master
git fetch origin master
git rebase -i origin/master
git status
git log
git push origin master
git remote add name https://github.com/dreamingbird88/kaggle/
git push --repo=name origin master
git rm another_test.txt 
git commit -m "Delete another_test.txt"
git log
git status
git push --repo=name origin master
git rm to_be_deleted.txt 
git status
git commit -m "delete to_be_deleted.txt"
git status
git log --oneline
git revert 17a722c
git log --oneline
git reset f0f78c4
git log --oneline
git reset b6cdb5c
git log --oneline
git status
git push --repo=name origin master
git push programming master 
git checkout bcbef3e configs/.bashrc
git checkout bcbef3e 
git revert bcbef3e
git reset bcbef3e

# list remote names
git remote -v

# Move a file
git mv MarkDown.md notes/
