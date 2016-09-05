#!/bin/sh

SESSIONNAME="home"

tmux has-session -t $SESSIONNAME > /dev/null 2>&1
if [ $? != 0 ]
then
  # start session
  tmux new-session -d -s $SESSIONNAME -n New

  # Window for new session
  tmux new-window -t $SESSIONNAME:1 -n Daily
  tmux send-keys 'vim ~/Data/Jeff/HomePage/每日事务.jeff' 'C-m'

  # Window for 每日事务
  # For vim-ipython. Deprecated.
  # tmux send-keys 'stty stop undef' 'C-m'
  # tmux select-pane -t 0
  # tmux split-window -h -p 40 -t $SESSIONNAME:Daily
  # tmux select-pane -t $SESSIONNAME:Daily.1
  # tmux send-keys 'ipython kernel' 'C-m'
  # tmux resize-pane -Z -t $SESSIONNAME:Daily.0

  # Default to the first window.
  tmux select-window -t $SESSIONNAME:New
fi

tmux attach -t $SESSIONNAME
