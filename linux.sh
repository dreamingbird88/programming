# apt install cannot handle multiple packages in one line.
sudo apt install vim
sudo apt install tmux
sudo apt install git

mkdir github
cd github
git clone https://github.com/dreamingbird88/programming

cd programming/configs
cp .vimrc .bashrc .tmux.config ~/

# Need to restart to make vim-slime effective.
git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
git clone https://github.com/jpalardy/vim-slime.git ~/.vim/bundle/vim-slime.vim

# Install python related.
sudo apt install ipython
sudo apt install python-pip
sudo apt install python-dev

sudo pip install pandas matplotlib scikit-learn

sudo -H pip install pandas
sudo pip install matplotlib
sudo pip install scikit-learn

pip install pandas

# TensorFlow
# https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#pip-installation
# Ubuntu/Linux 64-bit, CPU only:
sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl

# ipython
# import tensorflow as tf
# hello = tf.constant("Hello TensorFlow!")
# sess = tf.Session()
# print(sess.run(hello))






