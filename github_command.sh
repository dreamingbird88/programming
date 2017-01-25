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
repo_name="program"
git remote add ${repo_name} https://github.com/dreamingbird88/programming/
git remote set-url programming https://github.com/dreamingbird88/programming
# list remote names
git remote -v

# g4 sync
git pull https://github.com/dreamingbird88/programming/

# Add, delete or move a file
git add github_command.sh
git rm github_command.jeff
git mv MarkDown.md notes/

git commit -m "Update github_command"
git push -u ${repo_name} master

# 1. First create a repository with browser.
# 2. create a new repository on the command line
echo "# for_qi" >> README.md
git init
git add README.md
git commit -m "first commit"

git push -u ${repo_name} master
git push

git commit -a -m "python script combining passport photos for print"

# Download and refresh from another repository
git fetch for_qi master

# Delete a repository
# go to https://github.com/user_name/repository_name/setting
# dangerous zone --> delete ...
# local: rm -f -r dir_name

# check the commit (change list)
git status

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

# ---------------------------------------------------------------------------- #
cd ${githubdir}

is_new=true
project="ML4T_2017Spring/mc1p2_optimize_portfolio/optimization.py"
project="ML4T_2017Spring/mc1p1_assess_portfolio/analysis.py"

git_dest=$githubdir/mscs/${project}
mkdir -p  ${git_dest%/*}
qi_source=$qidir/temp/${project} 
if [ is_new ]; then
  qi_source=$qidir/${project} 
fi
cp -p ${qi_source} mscs/${project}
git add mscs/${project}

git commit -m "Modified mscs homework."
git push

vim ${qi_source} mscs/${project}
