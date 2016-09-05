#!/bin/sh

SESSIONNAME="work"

tmux has-session -t $SESSIONNAME > /dev/null 2>&1
if [ $? != 0 ]
then
  # Window for plusone
  tmux new-session -d -s $SESSIONNAME -n plusone
  tmux send-keys 'cd ~/clients/plusone/google3' 'C-m'
  tmux split-window -h -p 40 -t $SESSIONNAME:1
  tmux send-keys 'cd ~/clients/plusone/google3' 'C-m'
  tmux split-window -v -t $SESSIONNAME:1
  tmux send-keys 'cd ~/clients/plusone/google3' 'C-m'
  tmux select-pane -t 0

  # Window for gms-core
  tmux new-window -t $SESSIONNAME:2 -n gms-core
  tmux send-keys 'cd ~/clients/gms-core' 'C-m'
  tmux split-window -h -p 50 -t $SESSIONNAME:2
  tmux send-keys 'cd ~/clients/gms-core' 'C-m'
  tmux select-pane -t 0

  # Window for pplview
  tmux new-window -t $SESSIONNAME:3 -n pplview
  tmux send-keys 'cd ~/clients/pplview/google3' 'C-m'
  tmux split-window -h -p 40 -t $SESSIONNAME:3
  tmux send-keys 'cd ~/clients/pplview/google3' 'C-m'
  tmux split-window -v -t $SESSIONNAME:3
  tmux send-keys 'cd ~/clients/pplview/google3' 'C-m'
  tmux select-pane -t 0

  # Window for oncall
  tmux new-window -t $SESSIONNAME:4 -n oncall
  tmux send-keys 'cd /google/src/cloud/zhwang/zhwang-pplview/google3' 'C-m'
  tmux split-window -h -p 50 -t $SESSIONNAME:4
  tmux send-keys 'cd /google/src/cloud/zhwang/zhwang-pplview/google3' 'C-m'
  tmux select-pane -t 0

  # Window for pluskit
  tmux new-window -t $SESSIONNAME:5 -n pluskit
  tmux send-keys 'cd ~/clients/pluskit/google3' 'C-m'
  tmux split-window -h -p 50 -t $SESSIONNAME:5
  tmux send-keys 'cd ~/clients/pluskit/google3/experimental/users/zhwang/pluskit' 'C-m'
  tmux select-pane -t 0

  # Default to the first window.
  tmux select-window -t $SESSIONNAME:1
fi

tmux attach -t $SESSIONNAME

