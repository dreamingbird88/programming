#include <thread>
#include <iostream>
#include <sstream>
#include <vector>
#include <mutex>
#include <stdexcept>
#include <future>
#include <deque>
#include <map>

using namespace std;

const int MAXCAPACITY = 3;
mutex mx;
deque<string> data;
map<string,int> freq;
condition_variable room2produce, data2consume;

void print() {
    cout << "begin data" << endl;
    for(auto str : data)
    cout << str << endl;
    cout << "end data" << endl;
}

void consumer() {
    unique_lock<mutex> lck(mx);
    cout << "hi from thread " << this_thread::get_id() << endl;
    while(data.empty())
    data2consume.wait(lck);
    cout << "before consuming, data is as follows:" << endl;
    print();
    string line = data.front();
    data.pop_front();
    istringstream iss(line);
    string word;
    cout << "extracting words as: ";
    while( iss >> word) {
    cout << word << ",";
    freq[word]++;
    }
    cout << endl;
    cout << "after consuming, data is as follows:" << endl;
    print();
    room2produce.notify_one();
}

void producer(const string &line) {
    unique_lock<mutex> lck2(mx);
    cout << "hi from main:  " << this_thread::get_id() << endl;
    while(data.size()>=MAXCAPACITY)
    room2produce.wait(lck2);
    data.push_back(line);
    data2consume.notify_all();
}

int main() {
    int nThread = thread::hardware_concurrency();// maximum number of
threads hardware prefers
    vector<thread> ts;

    // launch consumers in threads
    for(int i=0;i<nThread;++i)
    ts.push_back(thread(consumer));

    // a single producer in main, since IO is bottleneck, no need for
concurrent read of input
    cout << "from main: " << endl;
    string line;
    while(getline(cin,line))
    producer(line);

    // wait consumers finish all work
    for(int i=0;i<nThread;++i)
    ts[i].join();

    //
    cout << "counts as following:" << endl;
    for(auto kvpair : freq)
    cout << kvpair.first << ": " << kvpair.second << endl;

}