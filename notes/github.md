# Notes on using github

### References:
* https://www.atlassian.com/git/tutorials/using-branches
* https://git-scm.com/docs
* https://training.github.com/kit/downloads/github-git-cheat-sheet.pdf

### Install and configure github 
* sudo apt-get install git
* git config --list
* git config --global user.email jeffnju@gmail.com
* git config --global user.name dreamingbird88
* sudo git config --system core.editor vim

### Clone a repository
* git clone kaggle **or** git clone https://github.com/dreamingbird88/kaggle
* git remote add name https://github.com/dreamingbird88/kaggle/

### Create a repository
1. First create a repository with browser;
2. Then clone it with command line.

### Edit/add files and push to the repository
* git add somefile.ext # *add* can refer to **modified**.
* git commit -m "a test." # **g4 change**
* git commit --amend
* git push -u origin master # **g4 submit**
* git push --repo=name origin master

### Others
* check the commit (change list): *git status* # **g4 pending**
* list remote names: *git remote -v* # **g4 list**
* move a file: *git mv MarkDown.md notes/*
* delete a file: *git rm to_be_deleted.txt*
* show commit log: *git log __--oneline__*
* sync with master: *git checkout __{commit_number}__ __{path}__* # **g4 sync**
* git revert bcbef3e
* git reset bcbef3e *or* git reset HEAD~2
* remove untracked files/directories from the working tree: *git clean -df*
* git rebase -i origin/master
* download objects and refs from another repository: *git fetch origin master* 
* vim .gitignore
