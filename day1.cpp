#include <utility>
using namespace std;
#include <iostream>
void explainPair(){
    pair<int,int>p={1,3};
    cout<<p.first<<" "<<p.second;
    pair<int,pair<int,int>> p={1,{1,2}};
    cout<<p.first<<" "<<p.second.first<<" "<<p.second.first;
}
