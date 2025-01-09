#include<bits/stdc++.h>
using namespace std;

bool binary_search_recursive(int arr[], int target, int low, int high) {
    if (low > high) return false; // Base case: target not found
    int mid = low + (high - low) / 2;
    if (arr[mid] == target) {
        return true; // Target found
    } else if (arr[mid] > target) {
        return binary_search_recursive(arr, target, low, mid - 1); // Search in left half
    } else {
        return binary_search_recursive(arr, target, mid + 1, high); // Search in right half
    }
}

int main() {
    ios::sync_with_stdio(0); cin.tie(0);
    int n; cin >> n;
    int x; cin >> x;

    int arr[n];
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }
    sort(arr, arr + n);

    if (binary_search_recursive(arr, x, 0, n - 1)) {
        cout << "Found\n";
    } else {
        cout << "Not Found\n";
    }
    return 0;
}

//TIME COMPLEXITY - O(logn)
//SPACE COMPLEXITY - O(logn)
