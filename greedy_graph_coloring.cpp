#include<bits/stdc++.h>
using namespace std;

void addEdge(vector<vector<int>>&graph,int u,int v){
    graph[u].push_back(v);
    graph[v].push_back(u);
}
bool is_adjacent(vector<vector<int>>&graph,int vertex,set<int>&s,int V){
    for(auto val : graph[vertex]){
        if(s.count(val)){
            return true;
        }
    }
    return false;

}

vector<set<int>>graphColoring(vector<vector<int>>&graph,int V){
    vector<int>c(V,0);
    vector<set<int>>ans;
    while(true){
        set<int>s;
        for(int i = 0;i<V;i++){
            if(!is_adjacent(graph,i,s,V)&&!c[i]){
                s.insert(i);
                c[i]=1;
            }
        }
        
        if(s.empty()){
            return ans;
        }
        ans.push_back(s);
    }

    return ans;

} 
int main(){
    ios::sync_with_stdio(0); cin.tie(0);
    int V;
    cin>>V;
    vector<vector<int>>graph(V);
    int E;
    cin>>E;
    for(int i = 0;i<E;i++){
        int x,y;
        cin>>x>>y;
        addEdge(graph,x-1,y-1);
    }
    vector<set<int>>ans = graphColoring(graph,V);
    cout<<ans.size()<<endl;
    for(auto element : ans){
        for(auto v : element){
            cout<<v+1<<' ';
        }
        cout<<endl;
    }

}
