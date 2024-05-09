#include <bits/stdc++.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <limits>
#include <vector>
using namespace std;
typedef vector<int> VI;
typedef vector<vector<int>> VVI;
typedef vector<string> VS;
typedef queue<int> QU;
typedef queue<pair<int, int>> QP;
typedef queue<pair<pair<int, int>, int>> QPP;
#define PB push_back
#define SZA(arr) (sizeof(arr) / sizeof(arr[0]))
#define SZ(x) ((int)x.size())
#define LEN(x) ((int)x.length())
#define REV(x) reverse(x.begin(), x.end());
#define trav(a, x) for (auto &a : x)
#define FOR(i, n) for (int i = 0; i < n; i++)
#define FOR_INNER(j, i, n) for (int j = i; j < n; j++)
#define FOR1(i, n) for (int i = 1; i <= n; i++)
#define SORT(x) sort(x.begin(), x.end())
#define MAX(x) *max_element(ALL(x))
#define MIN(x) *min_element(ALL(x))
#define SUM(x) accumulate(x.begin(), x.end(), 0LL)

// Short function start-->>
void printArray(int arr[], int length)
{
    for (int i = 0; i < length; ++i)
    {
        cout << arr[i] << " ";
    }
}
void printVector(vector<int> &arr)
{
    for (auto it : arr)
    {
        cout << it << " ";
    }
}
void printVectorString(vector<string> &arr)
{
    for (auto it : arr)
    {
        cout << it << endl;
    }
}
void printVectorVector(vector<vector<int>> x)
{
    for (const auto &row : x)
    {
        cout << "[";
        for (const auto &element : row)
        {
            cout << element << " ";
        }
        cout << "]";
        cout << std::endl;
    }
}
void printVectorVectorString(vector<vector<string>> x)
{
    for (const auto &row : x)
    {
        cout << "[";
        for (const auto &element : row)
        {
            cout << element << " ";
        }
        cout << "]";
        cout << std::endl;
    }
}
void printString(string s, int length)
{
    for (int i = 0; i < length; ++i)
    {
        cout << s[i] << " ";
    }
}
void printStack(stack<string> s)
{
    while (!s.empty())
    {
        cout << s.top() << " ";
        s.pop();
    }
    cout << endl;
}
void printAdjList(const vector<int> adj[], int V)
{
    for (int i = 0; i < V; ++i)
    {
        cout << "Adjacency list of vertex " << i << ": ";
        for (int j = 0; j < adj[i].size(); ++j)
        {
            cout << adj[i][j] << " ";
        }
        cout << endl;
    }
}

// Short function end-->>
/*
Li'll Interoduction----->>>
1️⃣ What is Dynamic Programing?
----> Dynamic Programming (DP) is a method used in mathematics and computer science to solve complex problems by breaking them down into simpler subproblems. By solving each subproblem only once and storing the results, it avoids redundant computations, leading to more efficient solutions for a wide range of problems.
2️⃣ The two common dynamic programming approaches are:
---->
Memoization: Known as the “top-down” dynamic programming, usually the problem is solved in the direction of the main problem to the base cases.
->Tend to store the value of subproblems in some map or table
Tabulation: Known as the “bottom-up '' dynamic programming, usually the problem is solved in the direction of solving the base cases to the main problem

3️⃣ How we are going to learn?
----> First try using recursion then to optimize we use memoization then we'll use tabulation for space optimise

4️⃣ How to convert Recursion ->Dynamic Programing?
----> 1. Declaring an array considering the size of the sub problems if n problem then its int dp[n+1]
      2. Storing the ans which is being computed for every sub problem
      3. Checking if the sub problem has been previously solved then the value will not be -1

$$$ RECURSION -> MEMOIZATION.
->i. Loook at the params changin ii. Before returning add it up iii. whenever we call recursion just check if it has been previously computed or not

$$$ MEMOIZATION -> TABULATION
->i.Chekc how much dp array is used then init it. ii.Look for the base case. iii. Try a loop iv. The change recursion code to dp v. At the end inside loop store in dp

5️⃣ How do you understand this is a dp problem.
----> i.Whenever the questions are like count the total no of ways.
ii. There're multiple ways to do this but you gotta tell me which is giving you a the minimal output or maximum output
For Recursion:
i.Try all possible ways like count or best way then you're trying to apply recursion
For Memoization:
you'll see recursaion having overlaping problem then you can use memo...
6️⃣ Shortcut trick
---->
i. Try to represent the problem in terms of index
ii. Do all possible stuffs on that index according to the problem statement
iii. If the qs says count all the ways ->sum up all the stuffs
    if says minimum-> take mini(all stuffs)
    if maxi-> take max(all stuffs)
7️⃣
---->
*/

/*
1. Fibonacci number
ANS :   A series of numbers in which each number ( Fibonacci number ) is the sum of the two preceding numbers.
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC : O(2^n)
// SC : O(n) due to the usage of the function call stack.
int fibonacciNumberRecur(int n)
{
    if (n <= 1)
        return n;
    return fibonacciNumberRecur(n - 1) + fibonacciNumberRecur(n - 2);
}

// Better ------Memoization----->
// TC : The overlapping subproblems will return the answer in constant time O(1). Therefore the total number of new subproblems we solve is ‘n’. Hence total time complexity is O(N).
// SC : We are using a recursion stack space(O(N)) and an array (again O(N)). Therefore total space complexity will be O(N) + O(N) ≈ O(N)
int fibonacciNumberMemo(int n, VI &dp)
{
    if (n <= 1)
        return n;
    // Check if prev solved
    if (dp[n] != -1)
        return dp[n];
    // Storing the ans which is being computed
    return dp[n] = fibonacciNumberMemo(n - 1, dp) + fibonacciNumberMemo(n - 2, dp);
}
// Optimal -----Tabulation----->
// TC : O(N) We are running a simple iterative loop
// SC : We are using an external array of size ‘n+1’. we're not using recirsion stack space
int fibonacciNumberTabu(int n, VI &dp)
{
    dp[0] = 0;
    dp[1] = 1;
    for (int i = 2; i <= n; i++)
        dp[i] = dp[i - 1] + dp[i - 2];
    return dp[n];
}
// Most Optimal -----Space Optimization----->
// TC : O(N) We are running a simple iterative loop
// SC : O(1)
int fibonacciNumberSpceOpti(int n)
{
    int prev2 = 0;
    int prev = 1;

    for (int i = 2; i <= n; i++)
    {
        int cur_i = prev2 + prev;
        prev2 = prev;
        prev = cur_i;
    }
    return prev;
}

/*
2. Climbing Stars/Count Ways To Reach The N-th Stairs
ANS : You have been given a number of stairs. Initially, you are at the 0th stair, and you need to reach the Nth stair.
Each time, you can climb either one step or two steps.
You are supposed to return the number of distinct ways you can climb from the 0th step to the Nth step.
** 1 <= 'T' <= 10
0 <= 'N' <= 10^5 **
You've to use mod
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC : O(2^n)
// SC : O(n) due to the usage of the function call stack.
int getCount(int currStep, int nStairs, const int &mod)
{

    // Base case.
    if (currStep >= nStairs)
    {

        return (currStep == nStairs);
    }

    //  Climb one stair.
    int oneStepcount = getCount(currStep + 1, nStairs, mod);

    //  Climb two stairs
    int twoStepCount = getCount(currStep + 2, nStairs, mod);

    return (oneStepcount + twoStepCount) % mod;
}
int countDistinctWaysRecr(int n)
{
    // Initialize the variable 'mod'.
    const int mod = 1000000007;

    // Initialize the variable 'ways'.
    int ways = getCount(0, n, mod);

    return ways;
}
// Better ------Memoization----->
// TC : The overlapping subproblems will return the answer in constant time O(1). Therefore the total number of new subproblems we solve is ‘n’. Hence total time complexity is O(N).
// SC : We are using a recursion stack space(O(N)) and an array (again O(N)). Therefore total space complexity will be O(N) + O(N) ≈ O(N)
int getCount(int currStep, int nStairs, vector<int> &dp, const int &mod)
{

    // Base case.
    if (currStep >= nStairs)
    {

        return (currStep == nStairs);
    }

    // Check we have already solution or not?.
    if (dp[currStep] != -1)
    {

        return dp[currStep];
    }

    // Climb one stair.
    int oneStepcount = getCount(currStep + 1, nStairs, dp, mod);

    // Climb two stairs.
    int twoStepCount = getCount(currStep + 2, nStairs, dp, mod);

    // Store for later use.
    dp[currStep] = (oneStepcount + twoStepCount) % mod;

    return dp[currStep];
}

int countDistinctWaysMemo(int n)
{

    // Initialize the variable 'mod'.
    const int mod = 1000000007;

    // Create an array 'dp' of length 'n + 1' with initial value '-1'.
    vector<int> dp(n + 1, -1);

    // Initialize the variable 'ways'.
    int ways = getCount(0, n, dp, mod);

    return ways;
}
// Optimal -----Tabulation----->
// TC : O(N) We are running a simple iterative loop
// SC : We are using an external array of size ‘n+1’. we're not using recirsion stack space
int countDistinctWaysTab(int n)
{

    // Initialize the variable 'mod'.
    const int mod = 1000000007;

    // Create an array 'dp' of length '2' with intial value '1'.
    vector<int> dp(2, 1);

    // Checking if 'n' is less than or equal to '1',
    // Because in that case there is no need for further calculation.
    if (n <= 1)
    {

        return dp[n];
    }

    // Iterate on the range '[2, n]'.
    for (int currStep = 2; currStep <= n; currStep++)
    {

        // Calculate ways to reach 'currStep'th stair.
        int currStepWays = (dp[0] + dp[1]) % mod;

        // Update 'dp' array.
        dp[0] = dp[1];

        dp[1] = currStepWays;
    }

    return dp[1];
}

// Most Optimal -----Space Optimization----->
// Time Complexity : O(log(N))
// Space complexity : O(log(N))
// Logic for Multiplication of Matrix 'F' and Matrix 'M'.
void multiply(int F[2][2], int M[2][2], const int &mod)
{

    int x = ((F[0][0] * 1LL * M[0][0]) % mod + (F[0][1] * 1LL * M[1][0]) % mod) % mod;
    int y = ((F[0][0] * 1LL * M[0][1]) % mod + (F[0][1] * 1LL * M[1][1]) % mod) % mod;
    int z = ((F[1][0] * 1LL * M[0][0]) % mod + (F[1][1] * 1LL * M[1][0]) % mod) % mod;
    int w = ((F[1][0] * 1LL * M[0][1]) % mod + (F[1][1] * 1LL * M[1][1]) % mod) % mod;

    F[0][0] = x;
    F[0][1] = y;
    F[1][0] = z;
    F[1][1] = w;
}

// Binary Matrix Exponentiation.
void power(int F[2][2], int nStairs, const int &mod)
{

    if (nStairs <= 1)
    {

        return;
    }

    int M[2][2] = {{1, 1}, {1, 0}};

    power(F, nStairs / 2, mod);

    multiply(F, F, mod);

    if (nStairs % 2 == 1)
    {

        multiply(F, M, mod);
    }
}

int fib(int nStairs, const int &mod)
{

    int F[2][2] = {{1, 1}, {1, 0}};

    // Base case.
    if (nStairs == 0)
    {

        return 0;
    }

    power(F, nStairs - 1, mod);

    return F[0][0];
}

int countDistinctWaysSOpti(int n)
{

    // Initialize the variable 'mod'.
    const int mod = 1000000007;

    // The no. of ways to climb the 'n' is equal to '(n + 1)th' Fibonacci Number.
    return fib(n + 1, mod);
}

/*
3.Frog Jump
ANS : There is a frog on the '1st' step of an 'N' stairs long staircase. The frog wants to reach the 'Nth' stair. 'HEIGHT[i]' is the height of the '(i+1)th' stair.If Frog jumps from 'ith' to 'jth' stair, the energy lost in the jump is given by absolute value of ( HEIGHT[i-1] - HEIGHT[j-1] ). If the Frog is on 'ith' staircase, he can jump either to '(i+1)th' stair or to '(i+2)th' stair. Your task is to find the minimum total energy used by the frog to reach from '1st' stair to 'Nth' stair.
Input :   || Output :
Intuition:
We're trying all possible ways so thats'y its we can think of gredy & recursion
why not we're using greedy algo?
-> The total energy required by the frog depends upon the path taken by the frog. If the frog just takes the cheapest path in every stage it can happen that it eventually takes a costlier path after a certain number of jumps. The following example will help to understand this.
[30,10,60,10,50,50] greedy sol=it only can go upto min(n-1,n-2)
so for that we'll get +20+0+40=60
but in no-greedy we'll get +30+0+10=40
Thats'y we're not using greedy algo

*/
// Bruteforce ------Recursion----->
// TC :O(2^N)
// SC :O(2^N)
int frogJumpRecur(int n, VI &heights)
{
    // Base case: when the frog reaches the first stair
    if (n == 1)
        return 0;

    // Initialize the variables to store energy loss for jumping to the left and right stairs
    int left = INT_MAX, right = INT_MAX;

    // Calculate energy loss for jumping to the left stair if possible
    if (n > 1)
        left = frogJumpRecur(n - 1, heights) + abs(heights[n - 1] - heights[n - 2]);

    // Calculate energy loss for jumping to the right stair if possible
    if (n > 2)
        right = frogJumpRecur(n - 2, heights) + abs(heights[n - 1] - heights[n - 3]);

    // Return the minimum energy loss between left and right jumps
    return min(left, right);
}

// Better ------Memoization----->
// There're overlaping sub-problems so the ans to the sub-problems will be similer thereby we can apply memoization
// Time Complexity: O(N)
// Reason: The overlapping subproblems will return the answer in constant time O(1). Therefore the total number of new subproblems we solve is ‘n’. Hence total time complexity is O(N).
// Space Complexity: O(N)
// Reason: We are using a recursion stack space(O(N)) and an array (again O(N)). Therefore total space complexity will be O(N) + O(N) ≈ O(N)
int frogJumpMemoHelper(int i, VI &heights, VI &dp)
{
    // Base case: when the frog reaches the first stair
    int n = SZ(heights);
    if (i == n - 1)
        return 0;
    if (dp[i] != -1)
        return dp[i];
    // Initialize the variables to store energy loss for jumping to the left and right stairs
    int oneJump = INT_MAX, twoJump = INT_MAX;

    // Calculate energy loss for jumping to the left stair if possible
    if (i + 1 < n)
    {
        oneJump = abs(heights[i] - heights[i + 1]) + frogJumpMemoHelper(i + 1, heights, dp);
    }

    if (i + 2 < n)
    {
        twoJump = abs(heights[i] - heights[i + 2]) + frogJumpMemoHelper(i + 2, heights, dp);
    }

    int ans = min(oneJump, twoJump);
    dp[i] = ans;
    return ans;
}
int frogJumpMemo(int n, VI &heights)
{
    VI dp(n + 1, -1);
    return frogJumpMemoHelper(0, heights, dp);
}
// Optimal -----Tabulation----->
// Time Complexity: O(N)
// Reason: We are running a simple iterative loop
// Space Complexity: O(N)
// Reason: We are using an external array of size ‘n+1’.
int frogJumpTabu(int n, VI &heights)
{
    VI dp(n, 0);
    dp[0] = 0;
    for (int i = 1; i < n; i++)
    {
        int fs = dp[i - 1] + abs(heights[i] - heights[i - 1]);
        int ss = INT_MAX;
        if (i > 1)
            ss = dp[i - 2] + abs(heights[i] - heights[i - 2]);
        dp[i] = min(fs, ss);
    }
    return dp[n - 1];
}
// Most Optimal -----Space Optimization----->
// Time Complexity: O(N)
// Reason: We are running a simple iterative loop
// Space Complexity: O(1)
// Reason: We are not using any extra space.
int frogJumpSOpti(int n, VI &height)
{
    int prev = 0;
    int prev2 = 0;
    for (int i = 1; i < n; i++)
    {
        int jumpTwo = INT_MAX;
        int jumpOne = prev + abs(height[i] - height[i - 1]);
        if (i > 1)
            jumpTwo = prev2 + abs(height[i] - height[i - 2]);

        int cur_i = min(jumpOne, jumpTwo);
        prev2 = prev;
        prev = cur_i;
    }
    return prev;
}

/*
4. Frog Jump with k Distances
ANS : There is an array of heights corresponding to 'n' stones. You have to reach from stone 1 to stone ‘n’.
From stone 'i', it is possible to reach stones 'i'+1, ‘i’+2… ‘i’+'k' , and the cost incurred will be | Height[i]-Height[j] |, where 'j' is the landing stone.
Return the minimum possible total cost incurred in reaching the stone ‘n’.
Input :   || Output :
*/
// Bruteforce ----------->
// TC : O(n * k), where n is the number of steps and k is the maximum number of steps backward.
// SC : Since the recursion depth can be at most n (the number of steps), the space complexity is O(n).
int minimizeCostRecr(int ind, int k, vector<int> &h)
{
    if (ind == 0)
        return 0;

    int minStep = INT_MAX;
    for (int j = 1; j <= k; j++)
    {
        if (ind - j >= 0)
        {
            int jump = minimizeCostRecr(ind - j, k, h) + abs(h[ind] - h[ind - j]);
            minStep = min(minStep, jump);
        }
        else
        {
            break;
        }
    }
    return minStep;
}

int minimizeCostR(int n, int k, vector<int> &h)
{
    return minimizeCostRecr(n - 1, k, h); // Start from the last index
}
// Better ------Memoization----->
// TC : O(N *K)
// SC : We are using a recursion stack space(O(N)) and an array (again O(N)). Therefore total space complexity will be O(N) + O(N) ≈ O(N)
int solveUtil(int ind, vector<int> &height, vector<int> &dp, int k)
{
    // Base case: If we are at the beginning (index 0), no cost is needed.
    if (ind == 0)
        return 0;

    // If the result for this index has been previously calculated, return it.
    if (dp[ind] != -1)
        return dp[ind];

    int mmSteps = INT_MAX;

    // Loop to try all possible jumps from '1' to 'k'
    for (int j = 1; j <= k; j++)
    {
        // Ensure that we do not jump beyond the beginning of the array
        if (ind - j >= 0)
        {
            // Calculate the cost for this jump and update mmSteps with the minimum cost
            int jump = solveUtil(ind - j, height, dp, k) + abs(height[ind] - height[ind - j]);
            mmSteps = min(jump, mmSteps);
        }
    }

    // Store the minimum cost for this index in the dp array and return it.
    return dp[ind] = mmSteps;
}

// Function to find the minimum cost to reach the end of the array
int minimizeCostMemo(int n, int k, vector<int> &height)
{
    vector<int> dp(n, -1);                  // Initialize a memoization array to store calculated results
    return solveUtil(n - 1, height, dp, k); // Start the recursion from the last index
}
// Optimal -----Tabulation----->
// TC : O(N*K)
// SC :O(N)
int minimizeCostTabHelper(int n, vector<int> &height, vector<int> &dp, int k)
{
    dp[0] = 0;

    // Loop through the array to fill in the dp array
    for (int i = 1; i < n; i++)
    {
        int mmSteps = INT_MAX;

        // Loop to try all possible jumps from '1' to 'k'
        for (int j = 1; j <= k; j++)
        {
            if (i - j >= 0)
            {
                int jump = dp[i - j] + abs(height[i] - height[i - j]);
                mmSteps = min(jump, mmSteps);
            }
        }
        dp[i] = mmSteps;
    }
    return dp[n - 1]; // The result is stored in the last element of dp
}

// Function to find the minimum cost to reach the end of the array
int minimizeCostTab(int n, int k, vector<int> &height)
{
    vector<int> dp(n, -1); // Initialize a memoization array to store calculated results
    return minimizeCostTabHelper(n, height, dp, k);
}
// Most Optimal -----Space Optimization----->
// TC : There is no space optimization approach cz space optimization takes long time to exeute
// SC :

/*
5. Maximum sum of non-adjacent elements
ANS : Given an array of ‘N’  positive integers, we need to return the maximum sum of the subsequence such that no two elements of the subsequence are adjacent elements in the array.
Note: A subsequence of an array is a list with elements of the array where some elements are deleted ( or not deleted at all) and the elements should be in the same order in the subsequence as in the array.

Leetcode qs:
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.
Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.
Input :  [1,2,4] || Output : pick 1+4=5
*/
// Bruteforce ----------->
// TC : O(2^n)
// SC :
// Try out all possible subsequences with the given condition which is pick the one with the minimum sum
/*
Intuitions:
As mentioned earlier we will use the pick/non-pick technique to generate all subsequences. We also need to take care of the non-adjacent elements in this step.
If we pick an element then, pick = arr[ind] + f(ind-2). The reason we are doing f(ind-2) is because we have picked the current index element so we need to pick a non-adjacent element so we choose the index ‘ind-2’ instead of ‘ind-1’.
Next we need to ignore the current element in our subsequence. So nonPick= 0 + f(ind-1). As we don’t pick the current element, we can consider the adjacent element in the subsequence.
*/
int generateSubsequences(int ind, VI &nums)
{
    // Base condition
    if (ind == 0)
        return nums[ind];
    if (ind < 0)
        return 0;
    int pick = nums[ind] + generateSubsequences(ind - 2, nums);
    int notPick = 0 + generateSubsequences(ind - 1, nums);
    return max(pick, notPick);
}
int robRecr(vector<int> &nums)
{
    int n = SZ(nums);
    return generateSubsequences(n, nums);
}
// Better ------Memoization----->
// Time Complexity: O(N)
// Reason: The overlapping subproblems will return the answer in constant time O(1). Therefore the total number of new subproblems we solve is ‘n’. Hence total time complexity is O(N).
// Space Complexity: O(N)
// Reason: We are using a recursion stack space(O(N)) and an array (again O(N)). Therefore total space complexity will be O(N) + O(N) ≈ O(N)
int memoHelper(int ind, VI &dp, VI &arr)
{
    // If the result for this index is already computed, return it
    if (dp[ind] != -1)
        return dp[ind];

    // Base cases
    if (ind == 0)
        return arr[ind];
    if (ind < 0)
        return 0;
    // Choose the current element or skip it, and take the maximum
    int pick = arr[ind] +
               memoHelper(ind - 2, dp, arr); // Choosing the current element
    int nonPick =
        0 + memoHelper(ind - 1, dp, arr); // Skipping the current element

    // Store the result in the DP table and return it
    return dp[ind] = max(pick, nonPick);
}
int robMemo(vector<int> &nums)
{
    int n = SZ(nums);
    VI dp(n, -1);
    return memoHelper(n - 1, dp, nums);
}
// Optimal -----Tabulation----->
// Time Complexity: O(N)
// Reason: We are running a simple iterative loop
// Space Complexity: O(N)
// Reason: We are using an external array of size ‘n+1’.
int tabuHelper(int n, vector<int> &arr, vector<int> &dp)
{
    // Base case: If there are no elements in the array, return 0
    dp[0] = arr[0];

    // Iterate through the elements of the array
    for (int i = 1; i < n; i++)
    {
        // Calculate the maximum value by either picking the current element
        // or not picking it (i.e., taking the maximum of dp[i-2] + arr[i] and dp[i-1])
        int pick = arr[i];
        if (i > 1)
            pick += dp[i - 2];
        int nonPick = dp[i - 1];

        // Store the maximum value in the dp array
        dp[i] = max(pick, nonPick);
    }

    // The last element of the dp array will contain the maximum sum
    return dp[n - 1];
}
int robTabu(vector<int> &nums)
{
    int n = SZ(nums);
    VI dp(n, 0);
    return tabuHelper(n, nums, dp);
}
// Most Optimal -----Space Optimization----->
// Time Complexity: O(N)
// Reason: We are running a simple iterative loop
// Space Complexity: O(1)
// Reason: We are not using any extra space.
/*
If we closely look at the values required at every iteration,

dp[i], dp[i-1], and  dp[i-2]

we see that for any i, we do need only the last two values in the array. So is there a need to maintain a whole array for it?

The answer is ‘No’. Let us call dp[i-1] as prev and dp[i-2] as prev2. Now understand the following illustration.
Each iteration’s cur_i and prev become the next iteration’s prev and prev2 respectively.
Therefore after calculating cur_i, if we update prev and prev2 according to the next step, we will always get the answer.
After the iterative loop has ended we can simply return prev as our answer.
*/
int robSopti(vector<int> &arr)
{
    int n = SZ(arr);
    int prev = arr[0]; // Initialize the maximum sum ending at the previous element
    int prev2 = 0;     // Initialize the maximum sum ending two elements ago

    for (int i = 1; i < n; i++)
    {
        int pick = arr[i]; // Maximum sum if we pick the current element
        if (i > 1)
            pick += prev2; // Add the maximum sum two elements ago

        int nonPick = 0 + prev; // Maximum sum if we don't pick the current element

        int cur_i = max(pick, nonPick); // Maximum sum ending at the current element
        prev2 = prev;                   // Update the maximum sum two elements ago
        prev = cur_i;                   // Update the maximum sum ending at the previous element
    }

    return prev; // Return the maximum sum
}
/*
6.House Robber II
ANS : A thief needs to rob money in a street. The houses in the street are arranged in a circular manner. Therefore the first and the last house are adjacent to each other. The security system in the street is such that if adjacent houses are robbed, the police will get notified.
Given an array of integers “Arr'' which represents money at each house, we need to return the maximum amount of money that the thief can rob without alerting the police.
Input :   || Output :
*/
// Bruteforce ----------->
// TC :
// SC :
// Better ------Memoization----->

// TC :
// SC :
// Optimal -----Tabulation----->

// TC :
// SC :
// Most Optimal -----Space Optimization----->
// Time Complexity: O(N )
// Reason: We are running a simple iterative loop, two times. Therefore total time complexity will be O(N) + O(N) ≈ O(N)
// Space Complexity: O(1)
// Reason: We are not using extra space.
int robII(vector<int> &nums)
{
    VI temp1, temp2;
    int n = SZ(nums);
    if (n == 1)
        return nums[0];
    FOR(i, n)
    {
        if (i != 0)
            temp1.PB(nums[i]);
        if (i != n - 1)
            temp2.PB(nums[i]);
    }
    return max(robSopti(temp1), robSopti(temp2));
}
/*
7.
ANS :
Input :   || Output :
*/
// Bruteforce ----------->
// TC :
// SC :
// Better ------Memoization----->

// TC :
// SC :
// Optimal -----Tabulation----->

// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :
/*
8.
ANS :
Input :   || Output :
*/
// Bruteforce ----------->
// TC :
// SC :
// Better ------Memoization----->

// TC :
// SC :
// Optimal -----Tabulation----->

// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :
/*
9.
ANS :
Input :   || Output :
*/
// Bruteforce ----------->
// TC :
// SC :
// Better ------Memoization----->

// TC :
// SC :
// Optimal -----Tabulation----->

// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :
/*
10.
ANS :
Input :   || Output :
*/
// Bruteforce ----------->
// TC :
// SC :
// Better ------Memoization----->

// TC :
// SC :
// Optimal -----Tabulation----->

// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :
/*
11.
ANS :
Input :   || Output :
*/
// Bruteforce ----------->
// TC :
// SC :
// Better ------Memoization----->

// TC :
// SC :
// Optimal -----Tabulation----->

// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :

/*
12.
ANS :
Input :   || Output :
*/
// Bruteforce ----------->
// TC :
// SC :
// Better ------Memoization----->

// TC :
// SC :
// Optimal -----Tabulation----->

// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :

/*
13.
ANS :
Input :   || Output :
*/
// Bruteforce ----------->
// TC :
// SC :
// Better ------Memoization----->

// TC :
// SC :
// Optimal -----Tabulation----->

// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :

/*
14.
ANS :
Input :   || Output :
*/
// Bruteforce ----------->
// TC :
// SC :
// Better ------Memoization----->

// TC :
// SC :
// Optimal -----Tabulation----->

// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :

/*
15.
ANS :
Input :   || Output :
*/
// Bruteforce ----------->
// TC :
// SC :
// Better ------Memoization----->

// TC :
// SC :
// Optimal -----Tabulation----->

// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :

// ================================MAIN START=================================>>
int main()
{
    /*
        Some short function
           int maxi = *max_element(arr.begin(), arr.end());
            int sum = accumulate(arr.begin(), arr.end(), 0);
    */

    // int n;
    // cout << "Enter the value of n: ";
    // cin >> n;
    // VI dp(n + 1, -1);
    // memset(dp,-1,sizeof dp);
    // cout << "The " << n << "th Fibonacci number is: " << fibonacciNumberRecur(n) << endl;
    // cout << "The " << n << "th Fibonacci number is: " << fibonacciNumberMemo(n, dp) << endl;
    // cout << "The " << n << "th Fibonacci number is: " << fibonacciNumberTabu(n, dp) << endl;
    // cout << "The " << n << "th Fibonacci number is: " << fibonacciNumberSpceOpti(n) << endl;
    // cout << "Recr " << countDistinctWaysRecr(5) << endl;
    // cout << "Memo " << countDistinctWaysMemo(5) << endl;
    // cout << "Tab " << countDistinctWaysTab(5) << endl;
    // cout << "S Opti " << countDistinctWaysSOpti(5) << endl;
    // VI h = {7, 2, 3, 6, 9, 6, 10, 10, 10, 3, 2, 7, 7, 4, 9, 5, 10, 5, 8, 7};
    // cout << "Recr " << frogJumpRecur(20, h) << endl;
    // cout << "Memo " << frogJumpMemo(20, h) << endl;
    // cout << "Tab " << frogJumpTabu(20, h) << endl;
    // cout << "S Opti " << frogJumpSOpti(20, h) << endl;
    // cout << "Recr " << minimizeCostR(20, 3, h) << endl;
    // cout << "Memo " << minimizeCostMemo(20, 3, h) << endl;
    // cout << "Tab " << minimizeCostTab(20, 3, h) << endl;
    VI h = {2, 7, 9, 3, 2};
    // cout << "Recr " << robRecr(h) << endl;
    // cout << "Memo " << robMemo(h) << endl;
    // cout << "Tab " << robTabu(h) << endl;
    // cout << "S opti " << robSopti(h) << endl;
    cout << "S opti " << robII(h) << endl;

    return 0;

    //  End code here-------->>

    return 0;
}
