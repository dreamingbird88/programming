# Install Bazel in local machine
# http://bazel.io/docs/install.html

cd $HOME
git clone https://github.com/google/bazel.git

sudo apt-get install openjdk-7-jdk openjdk-7-source
sudo apt-get install pkg-config zip g++ zlib1g-dev

cd bazel
./compile.sh
