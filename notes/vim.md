# Notes on learning/using vim

## Created a new filetype
```
:help new-filetype
```


## 汉字乱码 in .vimrc:
```
set fileencodings=utf-8,gbk,ucs-bom,cp936
```

## count words "n"
```
:%s///n
:%s/pattern//n
g + <Ctr+g>
```

# Normal Mode

## Range + Motion

Range | letter | line | cursors to end
--------- |---------------| -------------------- | ----
Command | lower | upper |  
insert front | i | I
insert back | a | A
substitute |  s  | S | 
change | c | cc | C == c$
re-do | u | U |
replace | r | | R
delete | | dd | D
copy | | Y == yy | 
upper case | gU | gUU | 
lower case | gu | guu |
change case | g~ | g~~ |



function | forward | backward
---------------| -------------------- | ----
insert letter | a | i
append line | o | O
paste | p | P
search in | f | F
search  before | t | T
follow | ; | ,
delete char | x | X 
command mode search | / | ?

within | whole
--- | ---
i | a
da) |  di)

# Compose reusable changes (T9)
e.g. delete a word
```
(1) daw (2) dbx (3) bdw
```
(1) is the best BECAUSE. reusable commands after above is
```
(1) daw (2) x (3) dw
```

# Don’t Count You Can Repeat (T11)
d7w is worse then dw...... but dw.... is worse then d5w

# number + operation: e.g.
```
 np # paste for buffer n times.
8@a # do operation in register "a" for 8 times.
```

## Display character under cursor
command | values
--- | ---
ga      | hex, ascii 
g8      | hex value of utf-8 
ggg?G   | rot13 whole file


## Moving cursors
* left down up right:  h j k l
* go display line: gj gk
* (line) start (unlike j k): n+ n- 
* (line) start end: nG n$
* (screen) top middle bottom: H M L
* (n line) below top or above bottom: nH nL
* first column (even space, different from ^) of line: 0 |
* first line in file: [[ 
* to column n: n|
* (block) start end: { }
* moving between {[()]} (T54): %
* (change list) forward backeward: g; g,
* (page) forward backward: <C+f> <C+b>
* (half page) forward backward: <C+d> <C+u>
* go to definition: gd
* go to the file: gf
* switch files between % and #: <C+^> 

## moving screen with cursor stays
* (page) top middle bottom: z<Enter> z. z-
* ? top middle bottome: nz<Enter> nz. nz-

## Marks & Jumps (T53)
```
:h mark-motions
:mark
:jumps
```
* create mark: m{a-zA-Z}
* go to mark line-beginning, exact position: 'm `m 
* position before the last jump within current file: `` 
* Location of last change: `.
* (last change or yank) start end: `[ `]
* (last visual selection) start end: `< `>
* jump forward backward: <C+i> <C+o>


## Buffers/Windows (T36)
:ls -- list files in the buffer
:bnext -- go to next file :bprevious :blast :bfirst
:buffer N -- buffer {bufferName}
:b! {N} leave a hidden file if current file is changed
:bdelete N1 N2 N3 -- :N, M bdelete -- 3,5 bdelete
:bufdo -- execute command for each buffer
:sbuffer N # open buffer N in horizontal window

## Change list
T56: change list    :h changelist
:changelist
Arguments/Windows    
T37: Argument List: arguments passed to VIM,    args can be changed by
:args **/*.cc  */*.txt
:args file1 file2
:argdo    :argadd    :argdel

:windo operates only for current tab page

T39: Split Windows:
Jump: <C-w>h   <C-w>j    <C-w>k  <C-w>l ;
<C-w> t left top <C-w> b right bottom
prev/next: <C-w><C-w>  <C-w><C-p>
    close: :only :close == :q but :close does not quit VIM
    open: <C+w>s    <C+w>v     :split  :vsplit
resize: <C-w>_  <C-w>| <C-w>= N<C-w>_  N<C-w>| <C-w>+ <C-w>-
<C-w>80|

Move windows
<C+w> K     <C+w> H      <C+w> J      <C+w> L    <C+w> r rotate the window
        
Tabs/Windows
T40: Tabs :h tabpage
    open :tabedit     :tabclose     :tabonly     :<C+w>T
    surf :tabnext gt     :tabprev gT    :tabnext {N} {N}gT
    move :tabmove {N} start from 0
    :tabs

T41: :edit %:h<Tab>
    % the filepath of the active file
    :h filepath without name    
Operator + Motion = Action (T12)
d2l -- daw -- dap[whole paragraph]
gUaw -- gUap
yf0 --
ndb -- 从当前行开始往前删除n字
:n,md   从第m行开始往前删除n行    
indent >>  or n>> indent; << or n<< remove indent

Simple calculation (T10 only for numbers)
subtract: 180<C-x>        add: 180<C-a>

Open command-line window: history of searches/Ex commands
q/ history of searches
q: history of Ex commands        

Insert Mode

Delete (T13) Just like linux terminal
<C-w> delete a word; <C-u> delete to line; <C-h> delete one character

T14: switch between normal mode
<C-[> back to normal mode
one time normal mode command <C-o>{normal mode command}: <C-o> zz


Insert unusual character
T17  
:h i_CTRL-V_digit
<C-v>{1234}    decimal    <C-v>u{00bf} hexadecimal    <C-v>{c1}{c2}: 1/2

T18: Unusual Characters by Digraph
:h digraphs-default :h digraph-table
    <C-k>12        

redirect output

Redirect the command mode output: redir
    :redir @{register}  or :redir > some_file  --- some command --- :redir END

Tip 43:
    :Explore or :E Open file explorer for the directory of the active buffer
        
T47:
    gj -- real line and display line
    word -- WORD: ewb ge -- EWB gE 大写的所有非空格字符        
Visual Mode -- for one-off changes

T19: Overwrite Existing Text with Replace Mode [from Normal Mode]
replace mode: R -- r
Virtual Replace mode: gR treat tab as multiple spaces:
to replace single character: r{char}, gr{char}
        
T21 Define a Visual Selection
character --  line-wise -- column wise v -- V -- <C+v>
last visual selected contents: gv
go to the other end of selection:  o
other {motions} h j k l; b e w; f{char}; $ ^ 0; n N

selection + motion (T24 usage of visual selection)
    Vr{character}:  replace the whole line to {character}
    <C-v> r|: replace selection with |
    gv: last visual mode selection

T25: <C-v> {motions} c: change selection

T26 from visual mode to insert mode: I -- A not i - a
<C-v> jj$ A: append at the end of lines
        
Command Mode

T27:
:[range]delete [x]    :[range]yank [x]    :[line]put [x]        :[range]join
:[line] copy {address}    :[range]move {address}
:[range]normal {commands}    
:[range]substitute/{pattern}/{string}/[flags]:
:[range]global/{pattern}/[cmd]

Selection + Offset
mix and match line numbers, marks, and patterns with offset (T28 p58)
line address --  :.,$p  
. current line; $ end of file; % start of file \< start of selection \> end of selection
offset -- :'<,'>p
pattern to select -- :/<html>/,/<\/html>/-p minus one line
plus 3 lines -- :.,.+3p
0 --- Virtual line above first line of the file
% --- The entire file (shorthand for :1,$)
'm --- Line containing mark m

Action + Position
T29    :t = :copy     :m = :move
:[range]copy {address} :6copy.
:copy == :t    
:6t. copy line 6 to current line
:t6 copy current line to below line 6
:'<,'>t0 copy selection to the start of file
:'<,'>m$ move selection to the end of file

Normal Mode command
T30 Run Normal Mode Commands Across a Range
:'<,'>normal {normal mode command}
:%normal A;
:%normal i//
one time command <C+o> {command}
        
T31 Repeat the Last Ex (linux) Command :@     
normal mode repeat .

T32 Tab-complete command: :buf<C+d>
    set wildmode=longest,list


Envioement
:set mouse=a  -- enable mouse
:set mouse-=a  -- disable mouse
:messages lots of past messages
:set paste    set paste!
:set all  打印所有选项
:set nooption  关闭option选项
:set nu   每行前打印行号
:set showmode  显示是输入模式还是替换模式
:set noic  查找时忽略大小写
:set ic   查找时忽略大小写
:set list  显示制表符(^I)和行尾符号
:set ts=8  为文本输入设置tab stops
:set window=n  设置文本窗口显示n行
:.=   打印当前行的行号
:=   打印文件中的行数
:l   使用字母"l"来显示许多的特殊字符，如制表符和换行符
^i(ctrl+i)或tab  插入文本时，插入移动的宽度，移动宽度是事先定义好的
:set ai   打开自动缩进
:set sw=n  将移动宽度设置为n个字符
:set number 显示行号
:set nonu 取消显示行号
选项设置
all：列出所有选项设置情况
term：设置终端类型
ignorance：在搜索中忽略大小写
list：显示制表位(Ctrl+I)和行尾标志（$)
number：显示行号
report：显示由面向行的命令修改过的数目
terse：显示简短的警告信息
warn：在转到别的文件时若没保存当前文件则显示NO write信息
nomagic：允许在搜索模式中，使用前面不带“\”的特殊字符
nowrapscan：禁止vi在搜索到达文件两端时，又从另一端开始
mesg：允许vi显示其他用户用write写到自己终端上的信息
:ab string1 string2 定义一个缩写，使得当插入string1时，用string2替换string1。当
要插入文本时，键入string1然后按Esc键，系统就插入了string2
:ab   显示所有缩写
:una string  取消string的缩写
:ab php                            " list abbreviations beginning with php
Shell command (T35)
Switch to shell environment temporarily
:shell  --- {shell envieroment} --- exit
Read/write from shell commands
:read !{cmd} read input buffer
space :write !{cmd} equivalent to :write ! {cmd} not equivalent to :write! sh # the 3rd one is a file name "sh"
:[range]write !{cmd}
==> 最后行方式命令
:n1,n2 co n3：将n1行到n2行之间的内容拷贝到第n3行下
:n1,n2 m n3：将n1行到n2行之间的内容移至到第n3行下
:n1,n2 d ：将n1行到n2行之间的内容删除
:e filename：打开文件filename进行编辑
:x 保存当前文件并退出
:n1,n2 w!command：将文件中n1行至n2行的内容作为command的输入并执行之，若不指定n1，n2，则表示将整个文件内容作为command的输入
:r!command：将命令command的输出结果放到当前行
:!command  执行shell的command命令，如:!ls
:!! 执行前一个shell命令
:r!command  读取command命令的输入并插入，如:r!ls会先执行ls，然后读入内容
:w!command  将当前已编辑文件作为command命令的标准输入并执行command命令，如:w!grep all
:cd directory  将当前工作目录更改为directory所表示的目录
:sh   将启动一个子shell，使用^d(ctrl+d)返回vi
:so file  在shell程序file中读入和执行命令
:<up> 上一命令; <down> 下一命令
:se<Up> 回到以前执行过的，以 "se" 开头的命令
:history / 如果你不愿用光标方向键，CTRL-P 作用就跟 <Up> 一样。而 CTRL-N 跟 <Down>
Read/write files
:r file   读入文件file内容，并插在当前行后
:w file -- write current content to file
:nr file  读入文件file内容，并插在第n行后


Mapping Keys/Command

Mapping command
Function:
" Vertical Split Buffer Function
function VerticalSplitBuffer(buffer)
   execute "vert belowright sb" a:buffer
endfunction
Mapping:
" Vertical Split Buffer Mapping
command -nargs=1 Vbuffer call VerticalSplitBuffer(<f-args>)
Running
:Vbuffer 1

Mapping Keys in VIM  :help map-which-keys
Part 1        Part 2        Part 3

{cmd} {attr} {lhs} {rhs}
{cmd}  is one of ':map', ':map!', ':nmap', ':vmap', ':imap',  ':cmap', ':smap', ':xmap', ':omap', ':lmap', etc.
{attr} is optional and one or more of the following: <buffer>, <silent>,       <expr> <script>, <unique> and <special>. More than one attribute can be specified to a map.
{lhs}  left hand side, is a sequence of one or more keys that are being       mapped.
{rhs}  right hand side, is the sequence of keys that the {lhs} keys are       mapped to.

maps the <F2> key to display the current date and time
:map <F2> :echo 'Current time is ' . strftime('%c')<CR>

maps the <F3> key to insert the current date and time in the current buffer:
:map! <F3> a<C-R>=strftime('%c')<CR><Esc>


==> 宏与缩写(避免使用控制键和符号，不要使用字符K、V、g、q、v、*、=和功能键)
:map key command_seq 定义一个键来运行command_seq，如:map e ea，无论什么时候都可以e移到一个字的末尾来追加文本
:map   在状态行显示所有已定义的宏
:umap key  删除该键的宏


==> mapping: [i,v,n]map, different mode mapping
:map <F7>  :'a,'bw file            " Write the lines from mark a to mark b to 'file'
:map <F8>  :.w file<CR>            " Write the current line to 'file'
:map <F9>  :r file                 " Read text from 'file' and insert it below the current line
:map <F10> :w<CR>:!php %<CR>       " Write the file and run it through php

:map \                             " list maps beginning with \
:nmap <Leader>s :source $MYVIMRC " source $MYVIMRC reloads the saved $MYVIMRC " <Leader> is \ by default, so those commands can be invoked by doing \v and \s
:nmap <Leader>v :e $MYVIMRC " opens $MYVIMRC for editing, or use :tabedit $MYVIMRC
noremap <Up> gk 或者 noremap <Down> gj "which would map the arrow keys to screen line movement instead of buffer line movement.

Remap the Caps Lock Key


Open files

Open a list of files
vim -p `cat yourlistoffiles` # open a tab for each file
vim -o one.txt two.txt three.txt # open a window for each file (total one tab)
vim -O # like -o but vertically
in the most common folder: vim -O2 `g4 p | tail -n +3 | head -n -1 | cut -d# -f1`

Open file at different position
vim +n filename: open file at line n
vi + filename: open file at last line
vi +/pattern filename: open file at first matched pattern
vi -r filename: recover filename

        
T50  search not limit to normal mode:
v then /{word} then {command} == {command}/{word}

T51:
vi} -- select contents within {}, same for vi", vi]
i -- inside    a -- all a} include the brace
} ) ] ' "         t <xml>tags</xml>
commands can be c y d
    visual mode select    v

Register 
Paste with register/variable/integer calculation
        
insert mode/command mode: <C+r>{register}
normal mode: “{register}
command mode check {register}:  :reg or :reg{register}
paste on content in insert mode -- similar to when in command mode (T15)
yf{char}: yank to register, default register 0:
<C-r>0 paste in Insert Mode while <C-r><C-p>{register} do not have indent
<C-r>{register}     <C-r>={variable}    <C-r>={integer calculation}
T33: <C-r><C-w> copy current word and place it in command line    
<C-r> <C-f> current file name in insert mode

T34 map <Up> to <C-p> in command mode: cnoremap <C-p> <Up>

ctrl-f Switch from Command-Line mode to the command-line window

T60 register :h quote{register}
"" unname register for x, s, d{motion}, c{motion}, and y{motion}
"0 - 9 y{motion}, lines
"- Last small delete
"a - z A-Z
"+ System Clipboard
"* Selection Registers
"% Name of the current file
"# Name of the alternate file
": Last Ex command
"/ Last search pattern
". Last insert content in the insert mode
"_ The black hole register
"= expression register: insert mode <C-r> =

T61 swap two words:
mm -- mark the place
    ve -- select the whole word
T62 :h linewise-register
    position at start or end: p -- gp     P -- gP

Macro Record
T64 - 66 Macro
    record: q{register} {actions} q
    use: @{register} or @@ last time; 10@{register} == 10000@{register} 宁多匆少
    series:     11@a
    parallel:  visual mode + :normal @a
T68 q{UpperCaseRegister}: append command to {LowerCaseRegister}
        
T69: different files
    :args
    :argdo normal @a    
    save changes to all open files---   :wall

T70:　每行加序号, use variables like bash
    :let i = 1
    qa
    I<c-r>=i<return>) <ESC>
    :let i += 1
    q
    jVG
    :normal @a
    

T71: edit macro paste it in editor
    paste it into the editor -- :put a ---     "ap    不一样，是否与原文同行
    yank it back: "add 比不上 0"ay$
    :let @a=substitute(@a, '\~', 'vU', 'g')
Vim Functions
:h function-list

Variables
:let a = "bar"
:let i = floor(3.4)
:echo &i
:echo &a


Search/Replace + Regular Expression
T72: case insensitive -- /\cjeff /\Cjeff    

T73: use \v to switch regular expression -- e.g. search "#3 or 6 digits/characters"
use “\” to accept patterns
\{4,5}
\+
\=
but not \*
    /#\([0-9a-fA-F]\{6}\|[0-9a-fA-F]\{3}\) or /\v#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})
    notice \v here make every character special except [0-9a-fA-F_]
\x -- [0-9a-fA-F]


Vim Regular Expression
:%s/\s\+#.\+//g
remove all comments

T74: Use the \V Literal Switch for Verbatim Searches, only / after it has special meaning
    e.g.: /a.k.a. can match ackwa but /\Va.k.a. only match a.k.a

T75: find duplicate words /\v<(\w+)\_s+\1>
\_s item matches whitespace or a line break
\1 refers to the first (), similar to \2 … \9. \0 is reserved for the whole

T76: Stake the Boundaries of a Word
    /\v<the>
    /\v%(And|D)rew Neil -- % skip the count in \1 … \9
    \w -- word characters \W -- except .


搜索及替换  :[range]s[ubstitute]/{pattern}/{string}/[flags] [count]
:%s/pattern/new/igc
三种 flags: g 全部, i 不分大小写, c 每次替换要确定
count 行数
range:
n1,n2 or n1, or , n2 默认缺省为光标所在行;
visual mode selection '<,'>
全文 %;
pattern
词的定界符 \< \>
^ 行的开头, $ 行的结尾
正则表达式 :%s/\<\(hey\|hi\)\>/hai/g
\(\) 限定    \| 表示或者    \<\> word delimeter
将行号加入每行 :%s/^/\=line(".") . ". "/g
\= 表示 表达式    line(".") 返回行号         ". " 表示该行内容
:4,$s/\d\+/\=submatch(0) + 1/ 第四行起每行第一个数字加1
\d\+ 表示数字 sequence
\=submatch(0)+1 表示第一个 matched 的 + 1
\=toupper(submatch(0)) 第一个 matched 的大写
:%s/\.\s*\w/\=toupper(submatch(0))/g
\.\s*\w 表示以 . 相隔 空格或tab的单词
:%s/\r//g   delete DOS Carriage Returns (^M == <C+v> <C+m>)
:'a,'bg/fred/s/dick/joe/gc : VERY USEFUL
:s/\(.*\):\(.*\)/\2 : \1/  reverse fields separated by :
:%s/^.\{-}pdf/new.pdf/     non-greedy matching \{-} to first pdf)
:s/fred/<c-r>a/g      substitute "fred" with contents of register "a"
:%s/^\(.*\)\n\1/\1$/      delete duplicate lines

# multiple commands
:%s/\f\+\.gif\>/\r&\r/g | v/\.gif$/d | %s/gif/jpg/
:%s/suck\|buck/loopy/gc       : ORing
:s/__date__/\=strftime("%c")/ : insert datestring
&   重复最后的:s命令
:g/text1/s/text2/text3 查找包含text1的行，用text3替换text2
:g/text/command  在所有包含text的行运行command所表示的命令
:v/text/command  在所有不包含text的行运行command所表示的命令

Q&A
stuck in the insert mode <Esc> <C+[> etc. are useless:
cause by <C+s> in insert mode
solution -- <C+q> quit it
Plugins


Plugin Manager -- Vbundle
sudo apt-get install git
git clone https://github.com/gmarik/Vundle.vim.git ~/.vim/bundle/Vundle.vim
add some lines in .vimrc

vim-slime
install github first
sudo apt-get install git-core

install in .vimrc
Plugin “jpalardy/vim-slime”
let g:slime_target = "tmux"
let g:slime_python_ipython = 1 “ for ipython copy past
<C+c> <C+c>
but can not change line

YouCompleteMe → to complex
git ….
git submodule update --init --recursive
install CMake via Software Center
cd ~/.vim/bundle/YouCompleteMe
./install.sh --clang-completer

vim-ipython -- deprecated
download github repository  -- https://github.com/ivanov/vim-ipython
copy files (including .vim and .py) to /home/jeff/.vim/ftplugin/python
install python-zmq via Software Center
each time
ipython kernel
stty stop undef
in vim: IPython
vim <leader> default == “\”
<F5> --- whole script
<C+s> -- one line

vimux -- vim interaction with tmux, deprecated
manual
Vimux-pyutils


index
= : (re)indent the text on the current line or on the area selected (SUPER)
=% : (re)indent the current braces { ... }
G=gg  : auto (re)indent entire document
vimdiff
用 vimdiff 显示文件差异, vim 会用垂直分割的方式打开两个文件
终端(不是在 Vim 中)      vimdiff main.c~ main.c
vim内部启动
:edit main.c
:vertical diffsplit main.c
:set noscrollbind 关闭滚动绑定(默认开启,不建议关闭)
更新差异 :diffupdate
消除差异
用 current 文件做参考复制修改 :dp 
用 other 文件做参考复制修改 :do
link two split files
:diffthis

// ================================================================= //
常用插件: http://blog.csdn.net/bokee/article/details/6633193

==> 安装 taglist,
1. 下载在 $HOME/.vim 里 unzip, 到 doc 子目录里去
2. 打开 vim, 执行:helptags .  (注意这个点号！)

==> 关联 ctag 与 taglist
1. 在shell中执行which ctags，记住返回结果，我的是/usr/bin/ctags    
2. 用vim打开.vim/plugin/taglist.vim文件，找到if!exist(“loaded_taglist”)这一行，在此行上面新加一行（大写O），加入如下内容： let list_Ctags_Cmd = "/usr/bin/ctags"（即将ctags的文件路径填到这里）
3. 切换到你希望查看的源文件的所在目录，执行ctags -R，即递归的检索当前目录下的所有源文件，并建立索引。

==> 在vim中执行:TlistToggle，可以发现在左侧出现了宏、函数、变量列表

==> 安装使用Cscope
0. 确认 Vim 是否支持 Cscope. 在 vim 下运行version 查看支持的特性，前面有前缀符号+的为支持
1. wget http://cscope.sourceforge.net/cscope_maps.vim -O ~/.vim/plugin/cscope_maps.vim
2. 在源代码目录 生成cscope数据库文件: cscope -Rbq -f path/xxx.out
3. 将cscope数据库载入Vim:  在 vim 下命令  cs add path/xxx.out  

==> 安装使用OmniCppComplete http://www.vim.org/scripts/script.php?script_id=1520
1. Unzip the plugin to ~/.vim
2. Run Vim and type the following command: :helptags $HOME/.vim/doc
3. Type :h omnicppcomplete and please read the installation paragraph.
4. 在~/.vimrc中加入以下两行：set nocp \n filetype plugin on  
5. 添加 STL 的自动完成 http://www.vim.org/scripts/script.php?script_id=2358





# References
* [Vim document](http://vimdoc.sourceforge.net/htmldoc/syntax.html)
* Practical VIM

