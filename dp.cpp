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
----> Dynamic Programming (DP) is a method used in mathematics and computer science to solve complex problems by breaking them down into simpler subproblems.
By solving each subproblem only once and storing the results, it avoids redundant computations,
 leading to more efficient solutions for a wide range of problems.
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
->i. Loook at the params changin
ii. Before returning add it up
iii. whenever we call recursion just check if it has been previously computed or not

$$$ MEMOIZATION -> TABULATION
->i.Check how much dp array is used then init it.
ii.Look for the base case.(Insteadof checking outer boundry first you can use i>0 for doing call)
iii. Try a loop
iv. The change recursion code to dp (Calls are going to according i,j)
v. At the end inside loop store in dp

5️⃣ How do you understand this is a dp problem.
----> i.Whenever the questions are like count the total no of ways.
ii. There're multiple ways to do this but you gotta tell me which is giving you a the minimal output or maximum output
For Recursion:
i.Try all possible ways like count or best way then you're trying to apply recursion
For Memoization:
you'll see recursaion having overlaping problem then you can use memo...

6️⃣ Shortcut trick for 1D DP or RECURSION******
---->
i. Try to represent the problem in terms of index
ii. Do all possible stuffs on that index according to the problem statement, Write base case and check for boundry
iii. If the qs says count all the ways ->sum up all the stuffs
    if says minimum-> take mini(all stuffs)
    if maxi-> take max(all stuffs)

7️⃣ Shortcut trick for 2D DP or RECURSION******
i. Express everything in terms of (row,col)
ii. Do all possible stuffs on that (row,col) according to the problem statement, Write base case and check for boundry
iii. If the qs says count all the ways ->sum up all the stuffs
    if says minimum-> take mini(all stuffs)
    if maxi-> take max(all stuffs)
---->

8️⃣Fixed starting point to variable ending point(at any last row) :  we generally tend to write the recursion from starting point

9️⃣Why we're not using Greedy Algorithm?
------>>Cz of Uniformity greedy always choose minimum elem but here sometimes we dont need minimum elem.
Example given in Frog Jump redirect to 413 line
*/

/*##############################1D DP#################################*/

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
Base Case:
If the frog is at the first stair (n == 1), no energy is required because the frog starts there. Therefore, frogJumpRecur(1) returns 0.
Recr :
Jump from n-1 stair: The energy required to jump from the n-1 stair to the n stair is the energy already spent to get to the n-1 stair plus the absolute difference in heights between stair n and stair n-1.
Jump from n-2 stair: The energy required to jump from the n-2 stair to the n stair is the energy already spent to get to the n-2 stair plus the absolute difference in heights between stair n and stair n-2.
The goal is to find the minimum energy between these two possible paths.
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
// Bruteforce ------Recursion----->
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
// Bruteforce ------Recursion----->
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
    return generateSubsequences(n - 1, nums);
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
// Bruteforce ------Recursion----->
// TC :
// SC :
// Better ------Memoization----->

// TC :
// SC :
// Optimal -----Tabulation----->

// TC :
// SC :
// Most Optimal -----Space Optimization----->
/*
Intuition : Houses arranged in a circular manner, meaning the first and last house are neighbors.
Two separate scenarios:
Robbing houses from the second house to the last house (excluding the first house).
Robbing houses from the first house to the second-to-last house (excluding the last house).

temp1 and temp2 are vectors to store the two subarrays.
If there is only one house (n == 1), return the money in that house because there is no constraint.
Iterate through the original array nums.
temp1 gets all elements of nums except the first element.
temp2 gets all elements of nums except the last element.
Compute the maximum money that can be robbed from temp1 and temp2.
Here we call Space OPtimized function cz this functions conditions is house are in a adj manner not circle so we can easily find maximum
Return the maximum of these two results.
*/

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

/*##############################2D/3D DP & DP ON GRIDS#################################*/

/*
7. Ninja's Training

ANS : A Ninja has an ‘N’ Day training schedule. He has to perform one of these three activities (Running, Fighting Practice, or Learning New Moves) each day. There are merit points associated with performing an activity each day. The same activity can’t be performed on two consecutive days.
We need to find the maximum merit points the ninja can attain in N Days.
We are given a 2D Array POINTS of size ‘N*3’ which tells us the merit point of specific activity on that particular day.
Our task is to calculate the maximum number of merit points that the ninja can earn.
Input :   || Output :
Why a Greedy Solution doesn’t work?
The first approach that comes to our mind is the greedy approach. We will see with an example how a greedy solution doesn’t give the correct solution.
We want to know the maximum amount of merit points. For the greedy approach, we will consider the maximum point activity each day,
respecting the condition that activity can’t be performed on consecutive days.
On Day 0, we will consider the activity with maximum points i.e 50.
On Day 1, the maximum point activity is 100 but we can’t perform the same activity in two consecutive days.
Therefore we will take the next maximum point activity of 11 points.
Total Merit points by Greedy Solution : 50+11 = 61
As this is a small example we can clearly see that we have a better approach, to consider activity with 10 points on day0 and 100 points on day1.
It gives us the total merit points as 110 which is better than the greedy solution.
So we see that the greedy solution restricts us from choices and we can lose activity with better points on the next day in the greedy solution. Therefore, it is better to try out all the possible choices as our next solution. We will use recursion to generate all the possible choices.
*/
// Bruteforce ------Recursion----->
/*
Intuition Behind the Code
Base Case:

The base case occurs when day == 0, meaning it's the first day. For the first day, you can choose any task except the task indexed by last (which is a task that might have been chosen the day before if it existed).
Iterate through all tasks (0, 1, 2), skipping the task equal to last, and find the maximum points you can get for the first day.
Recursive Case:

For each day (from n-1 to 0), iterate through each task (0, 1, 2) and recursively calculate the points for the current task plus the maximum points from the previous day, ensuring that the current task is not the same as the previous task (last).
Keep track of the maximum points obtained from these choices.
Recursive Helper Function:

maximumPointsRecrHelper(points, day, last) is the recursive helper function where day is the current day you are considering, and last is the task that was chosen the day before.
The function returns the maximum points that can be accumulated from day 0 to day, given that the task on day is not the same as last.
*/
// Time Complexity: O(N*4*3)
// Reason: There are N*4 states and for every state, we are running a for loop iterating three times.
// Space Complexity: O(N) + O(N*4)
// Reason: We are using a recursion stack space(O(N)) and a 2D array (again O(N*4)). Therefore total space complexity will be O(N) + O(N) ≈ O(N)

int maximumPointsRecrHelper(vector<vector<int>> &points, int day, int last)
{
    if (day == 0)
    {

        int maxi = 0;
        for (int i = 0; i <= 2; i++)
        {
            if (i != last)
                maxi = max(maxi, points[0][i]);
        }
        return maxi;
    }
    int maxi = 0;
    for (int i = 0; i <= 2; i++)
    {
        if (i != last)
        {
            int pts = points[day][i] + maximumPointsRecrHelper(points, day - 1, i);
            maxi = max(maxi, pts);
        }
    }
    return maxi;
}
int maximumPointsRecr(vector<vector<int>> &points, int n)
{
    return maximumPointsRecrHelper(points, n - 1, 3);
}
// Better ------Memoization----->
// Time Complexity: O(N*4*3)
// Reason: There are N*4 states and for every state, we are running a for loop iterating three times.
// Space Complexity: O(N) + O(N*4)
// Reason: We are using a recursion stack space(O(N)) and a 2D array (again O(N*4)). Therefore total space complexity will be O(N) + O(N) ≈ O(N)
int maximumPointsMemoHelper(vector<vector<int>> &points, int day, int last, VVI &dp)
{
    if (day == 0)
    {

        int maxi = 0;
        for (int i = 0; i <= 2; i++)
        {
            if (i != last)
                maxi = max(maxi, points[0][i]);
        }
        return maxi;
    }
    if (dp[day][last] != -1)
        return dp[day][last];
    int maxi = 0;
    for (int i = 0; i <= 2; i++)
    {
        if (i != last)
        {
            int pts = points[day][i] + maximumPointsMemoHelper(points, day - 1, i, dp);
            maxi = max(maxi, pts);
        }
    }
    return dp[day][last] = maxi;
}
int maximumPointsMemo(vector<vector<int>> &points, int n)
{
    VVI dp(n, VI(4, -1));
    return maximumPointsMemoHelper(points, n - 1, 3, dp);
}
// Optimal -----Tabulation----->
// Time Complexity: O(N*4*3)
// Reason: There are three nested loops
// Space Complexity: O(N*4)
// Reason: We are using an external array of size ‘N*4’.
int maximumPointsTabu(vector<vector<int>> &points, int n)
{
    // Create a 2D DP (Dynamic Programming) table to store the maximum points
    // dp[i][j] represents the maximum points at day i, considering the last activity as j
    vector<vector<int>> dp(n, vector<int>(4, 0));

    // Initialize the DP table for the first day (day 0)
    dp[0][0] = max(points[0][1], points[0][2]);
    dp[0][1] = max(points[0][0], points[0][2]);
    dp[0][2] = max(points[0][0], points[0][1]);
    dp[0][3] = max(points[0][0], max(points[0][1], points[0][2]));

    // Iterate through the days starting from day 1
    for (int day = 1; day < n; day++)
    {
        for (int last = 0; last < 4; last++)
        {
            dp[day][last] = 0;
            // Iterate through the tasks for the current day
            for (int task = 0; task <= 2; task++)
            {
                if (task != last)
                {
                    // Calculate the points for the current activity and add it to the
                    // maximum points obtained on the previous day (recursively calculated)
                    int activity = points[day][task] + dp[day - 1][task];
                    // Update the maximum points for the current day and last activity
                    dp[day][last] = max(dp[day][last], activity);
                }
            }
        }
    }

    // The maximum points for the last day with any activity can be found in dp[n-1][3]
    return dp[n - 1][3];
}
// Most Optimal -----Space Optimization----->
// Time Complexity: O(N*4*3)
// Reason: There are three nested loops
// Space Complexity: O(4)
// Reason: We are using an external array of size ‘4’ to store only one row.
int maximumPointsSopti(vector<vector<int>> &points, int n)
{
    // Initialize a vector to store the maximum points for the previous day's activities
    vector<int> prev(4, 0);

    // Initialize the DP table for the first day (day 0)
    prev[0] = max(points[0][1], points[0][2]);
    prev[1] = max(points[0][0], points[0][2]);
    prev[2] = max(points[0][0], points[0][1]);
    prev[3] = max(points[0][0], max(points[0][1], points[0][2]));

    // Iterate through the days starting from day 1
    for (int day = 1; day < n; day++)
    {
        // Create a temporary vector to store the maximum points for the current day's activities
        vector<int> temp(4, 0);
        for (int last = 0; last < 4; last++)
        {
            temp[last] = 0;
            // Iterate through the tasks for the current day
            for (int task = 0; task <= 2; task++)
            {
                if (task != last)
                {
                    // Calculate the points for the current activity and add it to the
                    // maximum points obtained on the previous day (stored in prev)
                    temp[last] = max(temp[last], points[day][task] + prev[task]);
                }
            }
        }
        // Update prev with the maximum points for the current day
        prev = temp;
    }

    // The maximum points for the last day with any activity can be found in prev[3]
    return prev[3];
}
/*
8.Unique Paths
ANS :  There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.
Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.
Input :   || Output :
*/
// Bruteforce -----Recursion------>
// TC : O(2xmxn)
// SC :O(path len)
// If question says all paths then we can try
int allPaths(int m, int n)
{
    if (m == 0 || n == 0)
        return 1;
    if (m < 0 || n < 0)
        return 0;
    return allPaths(m - 1, n) + allPaths(m, n - 1); // Recursive call to explore paths from up and left.
}
int uniquePathsRecr(int m, int n)
{
    if (m == 1 || n == 1) // Base case: If either dimension is 1, there is only one unique path.
        return 1;
    return uniquePathsRecr(m - 1, n) + uniquePathsRecr(m, n - 1); // Recursive call to explore paths from up and left.
}
// Better ------Memoization----->
// Time Complexity: O(M*N)
// Reason: At max, there will be M*N calls of recursion.
// Space Complexity: O((N-1)+(M-1)) + O(M*N)
// Reason: We are using a recursion stack space: O((N-1)+(M-1)), here (N-1)+(M-1) is the path length and an external DP Array of size ‘M*N’.
/*
if (m == 0 && n == 0) return 1;

Purpose: This base condition handles the case when the robot reaches the starting point (0, 0). There is exactly one way to be at the starting point, which is simply starting there.
Explanation: When m == 0 and n == 0, it means we are at the top-left corner. Since there is one unique way to be at the starting point (by starting there), we return 1.
if (m < 0 || n < 0) return 0;

Purpose: This base condition handles the boundaries of the grid.
Explanation: If the robot moves out of the grid's boundaries (i.e., to a negative index), there are no valid paths to the destination from there. Thus, we return 0 to indicate an invalid path.
*/
int uniquePathsMemoHelper(int i, int j, VVI &dp)
{
    // Base case: If we reach the top-left corner (0, 0), there is one way.
    if (i == 0 && j == 0)
        return 1;

    // If we go out of bounds or reach a blocked cell, there are no ways.
    if (i < 0 || j < 0)
        return 0;

    // If we have already computed the number of ways for this cell, return it.
    if (dp[i][j] != -1)
        return dp[i][j];

    // Calculate the number of ways by moving up and left recursively.
    int up = uniquePathsMemoHelper(i - 1, j, dp);
    int left = uniquePathsMemoHelper(i, j - 1, dp);

    // Store the result in the dp table and return it.
    return dp[i][j] = up + left;
}
int uniquePathsMemo(int m, int n)
{
    VVI dp(m, VI(n, -1));
    return uniquePathsMemoHelper(m - 1, n - 1, dp);
}
// Optimal -----Tabulation----->
// Time Complexity: O(M*N)
// Reason: There are two nested loops
// Space Complexity: O(M*N)
// Reason: We are using an external array of size ‘M*N’.
int uniquePathsTabuHelper(int m, int n, vector<vector<int>> &dp)
{
    // Loop through the grid using two nested loops
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            // Base condition: If we are at the top-left cell (0, 0), there is one way.
            if (i == 0 && j == 0)
            {
                dp[i][j] = 1;
                continue; // Skip the rest of the loop and continue with the next iteration.
            }

            // Initialize variables to store the number of ways from the cell above (up) and left (left).
            int up = 0;
            int left = 0;

            // If we are not at the first row (i > 0), update 'up' with the value from the cell above.
            if (i > 0)
                up = dp[i - 1][j];

            // If we are not at the first column (j > 0), update 'left' with the value from the cell to the left.
            if (j > 0)
                left = dp[i][j - 1];

            // Calculate the number of ways to reach the current cell by adding 'up' and 'left'.
            dp[i][j] = up + left;
        }
    }

    // The result is stored in the bottom-right cell (m-1, n-1).
    return dp[m - 1][n - 1];
}
int uniquePathsTabu(int m, int n)
{
    VVI dp(m, VI(n, -1));
    return uniquePathsTabuHelper(m, n, dp);
}
// Most Optimal -----Space Optimization----->
// Time Complexity: O(M*N)
// Reason: There are two nested loops
// Space Complexity: O(N)
// Reason: We are using an external array of size ‘N’ to store only one row.
// Objective: Reduce space complexity by using a single-dimensional array instead of a two-dimensional array.
// Strategy: Keep track of only the current row and the previous row's results.
int uniquePathsSopti(int m, int n)
{
    // Create a vector to represent the previous row of the grid.
    vector<int> prev(n, 0);

    // Iterate through the rows of the grid.
    for (int i = 0; i < m; i++)
    {
        // Create a temporary vector to represent the current row.
        vector<int> temp(n, 0);

        // Iterate through the columns of the grid.
        for (int j = 0; j < n; j++)
        {
            // Base case: If we are at the top-left cell (0, 0), there is one way.
            if (i == 0 && j == 0)
            {
                temp[j] = 1;
                continue;
            }

            // Initialize variables to store the number of ways from the cell above (up) and left (left).
            int up = 0;
            int left = 0;

            // If we are not at the first row (i > 0), update 'up' with the value from the previous row.
            if (i > 0)
                up = prev[j];

            // If we are not at the first column (j > 0), update 'left' with the value from the current row.
            if (j > 0)
                left = temp[j - 1];

            // Calculate the number of ways to reach the current cell by adding 'up' and 'left'.
            temp[j] = up + left;
        }

        // Update the previous row with the values calculated for the current row.
        prev = temp;
    }

    // The result is stored in the last cell of the previous row (n-1).
    return prev[n - 1];
}
/*
9. Unique Paths II
ANS : You are given an m x n integer array grid. There is a robot initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.
An obstacle and space are marked as 1 or 0 respectively in grid. A path that the robot takes cannot include any square that is an obstacle.
Return the number of possible unique paths that the robot can take to reach the bottom-right corner.

Input :   || Output :
*/
// Bruteforce -----Recursion------>
// TC : O(2xmxn)
// SC :O(path len)
// Here addition of a deadsell means one more base condition
/*
Why Different Starting Points:

Without Obstacles:
Starting from the bottom-right and moving to the top-left is straightforward for a simple grid without obstacles because every cell has exactly two choices (left or up) until it reaches the boundaries.
It leverages the simplicity of the problem where each cell's value depends only on its right and bottom neighbors.

With Obstacles:
Starting from the top-left and moving to the bottom-right is more intuitive when dealing with obstacles because it allows the function to stop and return immediately when an obstacle is encountered.
This approach aligns with the typical dynamic programming solution where you build the solution from the start and handle obstacles as you encounter them.
*/
// Its going one direction so its return 1 for big grid it will return accordingly
int uniquePathsWithObstaclesHelper(int m, int n, vector<vector<int>> &obstacleGrid)
{
    // Base case: if the current cell is an obstacle, return 0
    if (obstacleGrid[m][n] == 1)
        return 0;

    // Base case: if reached the bottom-right cell, return 1
    if (m == obstacleGrid.size() - 1 && n == obstacleGrid[0].size() - 1)
        return 1;

    int paths = 0;
    // Move right
    if (n + 1 < obstacleGrid[0].size())
        paths += uniquePathsWithObstaclesHelper(m, n + 1, obstacleGrid);
    // Move down
    if (m + 1 < obstacleGrid.size())
        paths += uniquePathsWithObstaclesHelper(m + 1, n, obstacleGrid);

    return paths;
}

int uniquePathsWithObstaclesRecr(vector<vector<int>> &arr)
{

    return uniquePathsWithObstaclesHelper(0, 0, arr);
}
// Better ------Memoization----->
// Time Complexity: O(N*M)
// Reason: At max, there will be N*M calls of recursion.
// Space Complexity: O((M-1)+(N-1)) + O(N*M)
// Reason: We are using a recursion stack space:O((M-1)+(N-1)), here (M-1)+(N-1) is the path length and an external DP Array of size ‘N*M’.
int mod = (int)(1e9 + 7);
int uniquePathsWithObstaclesMemoHelper(int i, int j, vector<vector<int>> &maze, vector<vector<int>> &dp)
{
    // Base cases
    if (i < 0 || j < 0 || maze[i][j] == 1)
        return 0; // If we go out of bounds or there's an obstacle at (i, j), return 0
    if (i == 0 && j == 0)
        return 1; // If we reach the destination (0, 0), return 1
    if (dp[i][j] != -1)
        return dp[i][j]; // If the result is already computed, return it

    // Recursive calls to explore paths from (i, j) to (0, 0)
    int up = uniquePathsWithObstaclesMemoHelper(i - 1, j, maze, dp);
    int left = uniquePathsWithObstaclesMemoHelper(i, j - 1, maze, dp);

    // Store the result in the DP table and return it
    return dp[i][j] = (up + left) % mod;
}
int uniquePathsWithObstaclesMemo(vector<vector<int>> &arr)
{
    int n = SZ(arr), m = SZ(arr[0]);
    VVI dp(n, VI(m, -1));
    return uniquePathsWithObstaclesMemoHelper(n - 1, m - 1, arr, dp);
}
// Optimal -----Tabulation----->
// Time Complexity: O(N*M)
// Reason: There are two nested loops
// Space Complexity: O(N*M)
// Reason: We are using an external array of size ‘N*M’’.
int uniquePathsWithObstaclesTabuHelper(int n, int m, vector<vector<int>> &maze,
                                       vector<vector<int>> &dp)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            // Base conditions
            if (maze[i][j] == 1)
            {
                dp[i][j] = 0; // If there's an obstacle at (i, j), no paths can pass through it
                continue;
            }
            if (i == 0 && j == 0)
            {
                dp[i][j] = 1; // If we are at the starting point, there is one path to it
                continue;
            }

            int up = 0;
            int left = 0;

            // Check if we can move up and left (if not at the edge of the maze)
            if (i > 0)
                up = dp[i - 1][j]; // Number of paths from above
            if (j > 0)
                left = dp[i][j - 1]; // Number of paths from the left

            // Total number of paths to reach (i, j) is the sum of paths from above and left
            dp[i][j] = (up + left) % mod;
        }
    }

    // The final result is stored in dp[n-1][m-1], which represents the destination
    return dp[n - 1][m - 1];
}
int uniquePathsWithObstaclesTabu(vector<vector<int>> &arr)
{
    int n = SZ(arr), m = SZ(arr[0]);
    VVI dp(n, VI(m, -1));
    return uniquePathsWithObstaclesTabuHelper(n, m, arr, dp);
}
// Most Optimal -----Space Optimization----->
// Time Complexity: O(M*N)
// Reason: There are two nested loops
// Space Complexity: O(N)
// Reason: We are using an external array of size ‘N’ to store only one row.

int uniquePathsWithObstaclesSopti(vector<vector<int>> &maze)
{
    int n = maze.size();    // Number of rows
    int m = maze[0].size(); // Number of columns

    vector<int> prev(m, 0); // Initialize a vector to store the previous row's path counts

    for (int i = 0; i < n; i++)
    {
        vector<int> temp(m, 0); // Initialize a temporary vector for the current row

        for (int j = 0; j < m; j++)
        {
            // Base conditions
            if (maze[i][j] == 1)
            {
                temp[j] = 0; // If there's an obstacle at (i, j), no paths can pass through it
                continue;
            }
            if (i == 0 && j == 0)
            {
                temp[j] = 1; // If we are at the starting point, there is one path to it
                continue;
            }

            int up = 0;
            int left = 0;

            // Check if we can move up and left (if not at the edge of the maze)
            if (i > 0)
                up = prev[j]; // Number of paths from above (previous row)
            if (j > 0)
                left = temp[j - 1]; // Number of paths from the left (current row)

            // Total number of paths to reach (i, j) is the sum of paths from above and left
            temp[j] = up + left;
        }

        prev = temp; // Update the previous row with the current row
    }

    // The final result is stored in prev[m - 1], which represents the destination in the last column
    return prev[m - 1];
}
/*
10. Minimum Path Sum In a Grid
ANS : Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.
Note: You can only move either down or right at any point in time.
Input :   || Output :
*/
// Bruteforce ------RECURSION----->
// TC :   O(2(mxn)) due to the exponential number of recursive calls in the worst case.
// SC : O(m+n) due to the depth of the recursion stack.
int minPathSumRecr(int m, int n, VVI &grid)
{
    // Base case :
    if (m == 0 && n == 0)
        return grid[0][0];
    if (m < 0 || n < 0)
        return 1e9;
    int up = grid[m][n] + minPathSumRecr(m - 1, n, grid);
    int left = grid[m][n] + minPathSumRecr(m, n - 1, grid);
    return min(up, left);
}
int minPathSum(vector<vector<int>> &grid)
{
    int m = SZ(grid);
    int n = SZ(grid[0]);
    return minPathSumRecr(m - 1, n - 1, grid);
}
// Better ------Memoization----->
// Time Complexity: O(N*M)
// Reason: At max, there will be N*M calls of recursion.
// Space Complexity: O((M-1)+(N-1)) + O(N*M)
// Reason: We are using a recursion stack space: O((M-1)+(N-1)), here (M-1)+(N-1) is the path length and an external DP Array of size ‘N*M’.
int minPathSumMemoH(int m, int n, VVI &grid, VVI &dp)
{
    // Base case :
    if (m == 0 && n == 0)
        return grid[0][0];
    if (m < 0 || n < 0)
        return 1e9;
    if (dp[m][n] != -1)
        return dp[m][n];
    int up = grid[m][n] + minPathSumMemoH(m - 1, n, grid, dp);
    int left = grid[m][n] + minPathSumMemoH(m, n - 1, grid, dp);
    return dp[m][n] = min(up, left);
}
int minPathSumMemo(vector<vector<int>> &grid)
{
    int m = SZ(grid);
    int n = SZ(grid[0]);
    VVI dp(m, VI(n, -1));
    return minPathSumMemoH(m - 1, n - 1, grid, dp);
}
// Optimal -----Tabulation----->
// Time Complexity: O(N*M)
// Reason: There are two nested loops
// Space Complexity: O(N*M)
// Reason: We are using an external array of size ‘N*M’’.

int minPathSumTabu(vector<vector<int>> &grid)
{

    int n = grid.size();
    int m = grid[0].size();
    VVI dp(n, VI(m, 0));
    FOR(i, n)
    {
        FOR(j, m)
        {
            if (i == 0 && j == 0)
                dp[i][j] =
                    grid[i][j]; // If we are at the top-left corner, the
                                // minimum path sum is the value at (0, 0)
            else
            {
                // Calculate the minimum path sum considering moving up and
                // moving left
                int up = grid[i][j];
                if (i > 0)
                    up += dp[i - 1]
                            [j]; // Include the minimum path sum from above
                else
                    up += 1e9; // A large value if moving up is not possible
                               // (out of bounds)

                int left = grid[i][j];
                if (j > 0)
                    left += dp[i][j - 1]; // Include the minimum path sum
                                          // from the left
                else
                    left += 1e9; // A large value if moving left is not
                                 // possible (out of bounds)

                // Store the minimum path sum in dp[i][j]
                dp[i][j] = min(up, left);
            }
        }
    }
    // The final result is stored in dp[n-1][m-1], which represents the
    // destination
    return dp[n - 1][m - 1];
}
// Most Optimal -----Space Optimization----->
// Time Complexity: O(M*N)
// Reason: There are two nested loops
// Space Complexity: O(N)
// Reason: We are using an external array of size ‘N’ to store only one row.
int minPathSumSO(VVI &grid)
{
    int n = grid.size();
    int m = grid[0].size();
    vector<int> prev(m, 0); // Initialize a vector to store the previous
                            // row's minimum path sums

    for (int i = 0; i < n; i++)
    {
        vector<int> temp(
            m, 0); // Initialize a temporary vector for the current row
        for (int j = 0; j < m; j++)
        {
            if (i == 0 && j == 0)
                temp[j] =
                    grid[i][j]; // If we are at the top-left corner, the
                                // minimum path sum is the value at (0, 0)
            else
            {
                // Calculate the minimum path sum considering moving up and
                // moving left
                int up = grid[i][j];
                if (i > 0)
                    up += prev[j]; // Include the minimum path sum from
                                   // above (previous row)
                else
                    up += 1e9; // A large value if moving up is not possible
                               // (out of bounds)

                int left = grid[i][j];
                if (j > 0)
                    left += temp[j - 1]; // Include the minimum path sum
                                         // from the left (current row)
                else
                    left += 1e9; // A large value if moving left is not
                                 // possible (out of bounds)

                // Store the minimum path sum in temp[j]
                temp[j] = min(up, left);
            }
        }
        prev = temp; // Update the previous row with the current row
    }

    // The final result is stored in prev[m-1], which represents the
    // destination in the last column
    return prev[m - 1];
}
/*
11. Minimum path sum in Triangular Grid
ANS : Given a triangle array, return the minimum path sum from top to bottom.
For each step, you may move to an adjacent number of the row below. More formally, if you are on index i on the current row, you may move to either index i or index i + 1 on the next row.
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC :O(2^n)
// SC :O(n) Recursion stack space
int minimumTotalRecr(int i, int j, VVI &tri)
{
    int n = SZ(tri);
    // Base case:
    if (i == n - 1)
        return tri[n - 1][j]; // If i in a last row
    int down = tri[i][j] + minimumTotalRecr(i + 1, j, tri);
    int dig = tri[i][j] + minimumTotalRecr(i + 1, j + 1, tri);
    return min(down, dig);
};
int minimumTotalR(vector<vector<int>> &tri)
{
    // RECURSION TC : SC :
    return minimumTotalRecr(0, 0, tri); // Here we don't know the ending
}
// Better ------Memoization----->
// Time Complexity: O(N*N)
// Reason: There are two nested loops
// Space Complexity: O(N*N)
// Reason: We are using an external array of size ‘N*N’. The stack space will be eliminated.
int minimumTotalMemo(int i, int j, VVI &tri, VVI &dp)
{
    int n = SZ(tri);
    // Base case:
    if (i == n - 1)
        return tri[n - 1][j]; // If i in a last row
    if (dp[i][j] != -1)
        return dp[i][j]; // If dp already haev calculated
    int down = tri[i][j] + minimumTotalMemo(i + 1, j, tri, dp);
    int dig = tri[i][j] + minimumTotalMemo(i + 1, j + 1, tri, dp);
    return dp[i][j] = min(down, dig);
}
int minimumTotalM(vector<vector<int>> &tri)
{

    // MEMOIZATION TC : SC :
    int n = SZ(tri);
    VVI dp(n, VI(n, -1));
    return minimumTotalMemo(0, 0, tri, dp);
}
// Optimal -----Tabulation----->
// Time Complexity: O(N*N)
// Reason: There are two nested loops
// Space Complexity: O(N*N)
// Reason: We are using an external array of size ‘N*N’. The stack space will be eliminated
int minimumTotalT(vector<vector<int>> &tri)
{

    // TABULATION TC : SC :
    int n = SZ(tri);
    VVI dp(n, VI(n, -1));

    // Initialize the bottom row of dp with the values from the triangle
    for (int j = 0; j < n; j++)
    {
        dp[n - 1][j] = tri[n - 1][j];
    }

    // Iterate through the tri rows in reverse order
    for (int i = n - 2; i >= 0; i--)
    {
        for (int j = i; j >= 0; j--)
        {
            // Calculate the minimum path sum for the current cell
            int down = tri[i][j] + dp[i + 1][j];
            int diagonal = tri[i][j] + dp[i + 1][j + 1];

            // Store the minimum of the two possible paths in dp
            dp[i][j] = min(down, diagonal);
        }
    }

    // The top-left cell of dp now contains the minimum path sum
    return dp[0][0];
}
// Most Optimal -----Space Optimization----->
// Time Complexity: O(N*N)
// Reason: There are two nested loops
// Space Complexity: O(N)
// Reason: We are using an external array of size ‘N’ to store only one row.
int minimumTotalSO(vector<vector<int>> &tri)
{

    int n = SZ(tri);
    // Create two arrays to store the current and previous row values
    vector<int> front(n, 0); // Represents the previous row
    vector<int> cur(n, 0);   // Represents the current row

    // Initialize the front array with values from the last row of the triangle
    for (int j = 0; j < n; j++)
    {
        front[j] = tri[n - 1][j];
    }

    // Iterate through the tri rows in reverse order
    for (int i = n - 2; i >= 0; i--)
    {
        for (int j = i; j >= 0; j--)
        {
            // Calculate the minimum path sum for the current cell
            int down = tri[i][j] + front[j];
            int diagonal = tri[i][j] + front[j + 1];

            // Store the minimum of the two possible paths in the current row
            cur[j] = min(down, diagonal);
        }
        // Update the front array with the values from the current row
        front = cur;
    }

    // The front array now contains the minimum path sum from the top to the bottom of the triangle
    return front[0];
}

/*
12. Minimum Falling Path Sum
ANS : Given an n x n array of integers matrix, return the minimum sum of any falling path through matrix.

A falling path starts at any element in the first row and chooses the element in the next row that is either directly below or diagonally left/right. Specifically, the next element from position (row, col) will be (row + 1, col - 1), (row + 1, col), or (row + 1, col + 1).
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC : O(3^n)~ exponential in nature SC : O(N)
int minFallingPathSumRecr(int i, int j, VVI &mat)
{
    int m = SZ(mat[0]);
    // Base Case :
    if (j < 0 || j >= m)
        return 1e9; // Boundry check
    if (i == 0)
        return mat[0][j]; // If you reached at 1st col then return jth value

    int up = mat[i][j] + minFallingPathSumRecr(i - 1, j, mat);
    int dl = mat[i][j] + minFallingPathSumRecr(i - 1, j - 1, mat);
    int dr = mat[i][j] + minFallingPathSumRecr(i - 1, j + 1, mat);
    return min(up, min(dl, dr));
}
int minFallingPathSumR(vector<vector<int>> &mat)
{
    int n = SZ(mat[0]);
    int pathSum = INT_MAX;
    FOR(j, n)
    {
        pathSum = min(minFallingPathSumRecr(n - 1, j, mat), pathSum);
    }
    return pathSum;
}
// Better ------Memoization----->
// Time Complexity: O(N*N)
// Reason: At max, there will be M*N calls of recursion to solve a new problem,
// Space Complexity: O(N) + O(N*M)
// Reason: We are using a recursion stack space: O(N), where N is the path length and an external DP Array of size ‘N*M’.
int minFallingPathSumMemo(int i, int j, VVI &mat, VVI &dp)
{
    int m = SZ(mat[0]);
    // Base Case :
    if (j < 0 || j >= m)
        return 1e9; // Boundry check
    if (i == 0)
        return mat[i][j]; // If you reached at 1st col then return jth value
    if (dp[i][j] != -1)
        return dp[i][j];
    int up = mat[i][j] + minFallingPathSumMemo(i - 1, j, mat, dp);
    int dl = mat[i][j] + minFallingPathSumMemo(i - 1, j - 1, mat, dp);
    int dr = mat[i][j] + minFallingPathSumMemo(i - 1, j + 1, mat, dp);
    return dp[i][j] = min(up, min(dl, dr));
}
int minFallingPathSumM(vector<vector<int>> &mat)
{
    int n = SZ(mat);
    int m = SZ(mat[0]);
    VVI dp(n, VI(m, -1));
    int pathSum = 1e9;
    FOR(j, n)
    pathSum = min(minFallingPathSumMemo(n - 1, j, mat, dp), pathSum);
    return pathSum;
}
// Optimal -----Tabulation----->
// Time Complexity: O(N*M)
// Reason: There are two nested loops
// Space Complexity: O(N*M)
// Reason: We are using an external array of size ‘N*M’. The stack space will be eliminated.
int minFallingPathSumT(vector<vector<int>> &mat)
{
    int n = SZ(mat);
    int m = SZ(mat[0]);
    VVI dp(n, VI(m, 0));

    // Initialize the first row of dp with values from the matrix (base condition)
    FOR(j, m)
    dp[0][j] = mat[0][j];
    // Iterate through the matrix rows starting from the second row
    for (int i = 1; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            // Calculate the maximum path sum for the current cell considering three possible directions: up, left diagonal, and right diagonal

            // Up direction
            int up = mat[i][j] + dp[i - 1][j];

            // Left diagonal direction (if it's a valid move)
            int leftDiagonal = mat[i][j];
            if (j - 1 >= 0)
            {
                leftDiagonal += dp[i - 1][j - 1];
            }
            else
            {
                leftDiagonal += 1e9; // A very large negative value to represent an invalid path
            }

            // Right diagonal direction (if it's a valid move)
            int rightDiagonal = mat[i][j];
            if (j + 1 < m)
            {
                rightDiagonal += dp[i - 1][j + 1];
            }
            else
            {
                rightDiagonal += 1e9; // A very large negative value to represent an invalid path
            }

            // Store the minimum of the three paths in dp
            dp[i][j] = min(up, min(leftDiagonal, rightDiagonal));
        }
    }

    // Find the minimum value in the last row of dp, which represents the minimum path sums ending at each cell
    int mini = INT_MAX;
    for (int j = 0; j < m; j++)
    {
        mini = min(mini, dp[n - 1][j]);
    }

    // The minimum path sum is the minimum value in the last row
    return mini;
}
// Most Optimal -----Space Optimization----->
// Time Complexity: O(N*M)
// Reason: There are two nested loops
// Space Complexity: O(M)
// Reason: We are using an external array of size ‘M’ to store only one row.
int minFallingPathSumSO(vector<vector<int>> &matrix)
{
    int n = matrix.size();    // Number of rows in the matrix
    int m = matrix[0].size(); // Number of columns in the matrix

    vector<int> prev(m, 0); // Represents the previous row's maximum path sums
    vector<int> cur(m, 0);  // Represents the current row's maximum path sums

    // Initialize the first row (base condition)
    for (int j = 0; j < m; j++)
    {
        prev[j] = matrix[0][j];
    }

    for (int i = 1; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            // Calculate the maximum path sum for the current cell considering three possible directions: up, left diagonal, and right diagonal

            // Up direction
            int up = matrix[i][j] + prev[j];

            // Left diagonal direction (if it's a valid move)
            int leftDiagonal = matrix[i][j];
            if (j - 1 >= 0)
            {
                leftDiagonal += prev[j - 1];
            }
            else
            {
                leftDiagonal += 1e9; // A very large negative value to represent an invalid path
            }

            // Right diagonal direction (if it's a valid move)
            int rightDiagonal = matrix[i][j];
            if (j + 1 < m)
            {
                rightDiagonal += prev[j + 1];
            }
            else
            {
                rightDiagonal += 1e9; // A very large negative value to represent an invalid path
            }

            // Store the maximum of the three paths in the current row
            cur[j] = min(up, min(leftDiagonal, rightDiagonal));
        }

        // Update the 'prev' array with the values from the 'cur' array for the next iteration
        prev = cur;
    }

    // Find the maximum value in the last row of 'prev', which represents the maximum path sums ending at each cell
    int maxi = INT_MAX;
    for (int j = 0; j < m; j++)
    {
        maxi = min(maxi, prev[j]);
    }

    // The maximum path sum is the maximum value in the last row of 'prev'
    return maxi;
}
/*
13.  Ninja and his friends 3D DP(Cherry Pickup)
ANS : We are given an ‘N*M’ matrix. Every cell of the matrix has some chocolates on it, mat[i][j] gives us the number of chocolates. We have two friends ‘Alice’ and ‘Bob’. initially, Alice is standing on the cell(0,0) and Bob is standing on the cell(0, M-1). Both of them can move only to the cells below them in these three directions: to the bottom cell (↓), to the bottom-right cell(↘), or to the bottom-left cell(↙).

When Alica and Bob visit a cell, they take all the chocolates from that cell with them. It can happen that they visit the same cell, in that case, the chocolates need to be considered only once.

They cannot go out of the boundary of the given matrix, we need to return the maximum number of chocolates that Bob and Alice can together collect.
Input :   || Output :
*/
// Bruteforce ------Recursion----->

/*
Intuition : After read the question we've to write 2 recursion call  for bob and alice then sum it up then you've to trace the path then you've to subtract the path if there is anything common so it will longer time,
so we can merge it and make it single.
Here're rules -
i. Express everthing in terms of Bob(i1,j1) & Alice(i2,j2)
ii. Explore all the paths
iii. Return maximum sum
Observed : Fixed starting point to variable ending point(at any last row) :  we generally tend to write the recursion from starting point
Observed : We can ommit any of i1,i2 cz bob & alice only go to his 2nd row in same time
*/
// TC :O(3^nx3^n)~exponential
// SC :O(n)
int chocoPickRecr(int i, int j1, int j2, VVI &grid)
{
    int n = SZ(grid);
    int m = SZ(grid[0]);
    // First Base case boundry check:
    if (j1 < 0 || j1 >= m || j2 < 0 || j2 >= m)
        return -1e8;
    // Second base case :
    //  When both reached last row
    if (i == n - 1)
    {
        if (j1 == j2)
            return grid[i][j1]; // Both reached in a same col
        else
            return grid[i][j1] + grid[i][j2]; // Diffrent
    }
    // Explore all the path bob and alice go together
    // If one movement of bob, alice have all remain movement
    int maxi = INT_MIN;
    // Try all possible moves (up, left, right) for both positions (j1, j2)
    for (int dj1 = -1; dj1 <= 1; dj1++)
    {
        for (int dj2 = -1; dj2 <= 1; dj2++)
        {
            if (j1 == j2)
                maxi = max(maxi, grid[i][j1] + chocoPickRecr(i + 1, j1 + dj1, j2 + dj2, grid)); // Both in same col
            else
                maxi = max(maxi, grid[i][j1] + grid[i][j2] + chocoPickRecr(i + 1, j1 + dj1, j2 + dj2, grid)); // Diffrent
        }
    }
    return maxi;
}
int maximumChocolatesR(vector<vector<int>> &grid)
{
    int m = SZ(grid[0]);
    return chocoPickRecr(0, 0, m - 1, grid); // bob and alice going simultaneously bellow his row
}
// Better ------Memoization----->
// Time Complexity: O(N*M*M) * 9
// Reason: At max, there will be N*M*M calls of recursion to solve a new problem and in every call, two nested loops together run for 9 times.
// Space Complexity: O(N) + O(N*M*M)
// Reason: We are using a recursion stack space: O(N), where N is the path length and an external DP Array of size ‘N*M*M’.
int chocoPickMemo(int i, int j1, int j2, VVI &grid, vector<vector<vector<int>>> &dp)
{
    int n = SZ(grid);
    int m = SZ(grid[0]);
    // First Base case boundry check:
    if (j1 < 0 || j1 >= m || j2 < 0 || j2 >= m)
        return -1e8;
    // Memo base case :
    if (dp[i][j1][j2] != -1)
        return dp[i][j1][j2];
    // Second base case :
    //  When both reached last row
    if (i == n - 1)
    {
        if (j1 == j2)
            return grid[i][j1]; // Both reached in a same col
        else
            return grid[i][j1] + grid[i][j2]; // Diffrent
    }
    // Explore all the path bob and alice go together
    // If one movement of bob, alice have all remain movement
    int maxi = INT_MIN;
    // Try all possible moves (up, left, right) for both positions (j1, j2)
    for (int dj1 = -1; dj1 <= 1; dj1++)
    {
        for (int dj2 = -1; dj2 <= 1; dj2++)
        {
            if (j1 == j2)
                maxi = max(maxi, grid[i][j1] + chocoPickMemo(i + 1, j1 + dj1, j2 + dj2, grid, dp)); // Both in same col
            else
                maxi = max(maxi, grid[i][j1] + grid[i][j2] + chocoPickMemo(i + 1, j1 + dj1, j2 + dj2, grid, dp)); // Diffrent
        }
    }
    return dp[i][j1][j2] = maxi;
}
int maximumChocolatesM(vector<vector<int>> &grid)
{
    int n = SZ(grid);
    int m = SZ(grid[0]);
    vector<vector<vector<int>>> dp(n, vector<vector<int>>(m, vector<int>(m, -1)));
    return chocoPickMemo(0, 0, m - 1, grid, dp); // bob and alice going simultaneously bellow his row
}
// Optimal -----Tabulation----->
// Time Complexity: O(N*M*M)*9
// Reason: The outer nested loops run for (N*M*M) times and the inner two nested loops run for 9 times.
// Space Complexity: O(N*M*M)
// Reason: We are using an external array of size ‘N*M*M’. The stack space will be eliminated.
int maximumChocolatesT(vector<vector<int>> &grid)
{
    int n = SZ(grid);
    int m = SZ(grid[0]);
    vector<vector<vector<int>>> dp(n, vector<vector<int>>(m, vector<int>(m, 0)));
    // In the recursive code, our base condition is when we reach the last row, therefore in our dp array, we will also initialize dp[n - 1][][], i.e(the last plane of 3D Array) as the base condition.Dp[n - 1][j1][j2] means Alice is at(n - 1, j1) and Bob is at(n - 1, j2).As this is the last row, its value will be equal to mat[i][j1], if (j1 == j2) and mat[i][j1] + mat[i][j2] otherwise.
    // Base case :
    // Initialize the DP array for the last row
    for (int j1 = 0; j1 < m; j1++)
    {
        for (int j2 = 0; j2 < m; j2++)
        {
            if (j1 == j2)
                dp[n - 1][j1][j2] = grid[n - 1][j1];
            else
                dp[n - 1][j1][j2] = grid[n - 1][j1] + grid[n - 1][j2];
        }
    }
    // Outer nested loops for traversing the DP array from the second-to-last row up to the first row
    for (int i = n - 2; i >= 0; i--)
    {
        for (int j1 = 0; j1 < m; j1++)
        {
            for (int j2 = 0; j2 < m; j2++)
            {
                int maxi = INT_MIN;

                // Inner nested loops to try out 9 options (diagonal moves)
                for (int di = -1; di <= 1; di++)
                {
                    for (int dj = -1; dj <= 1; dj++)
                    {
                        int ans;

                        if (j1 == j2)
                            ans = grid[i][j1];
                        else
                            ans = grid[i][j1] + grid[i][j2];

                        // Check if the move is valid (within the grid boundaries)
                        if ((j1 + di < 0 || j1 + di >= m) || (j2 + dj < 0 || j2 + dj >= m))
                            ans += -1e9; // A very large negative value to represent an invalid move
                        else
                            ans += dp[i + 1][j1 + di][j2 + dj]; // Include the DP result from the next row

                        maxi = max(ans, maxi); // Update the maximum result
                    }
                }
                dp[i][j1][j2] = maxi; // Store the maximum result for this state in the DP array
            }
        }
    }

    // The maximum chocolates that can be collected is stored at the top-left corner of the DP array
    return dp[0][0][m - 1];
}
// Most Optimal -----Space Optimization----->
// Time Complexity: O(N*M*M)*9
// Reason: The outer nested loops run for (N*M*M) times and the inner two nested loops run for 9 times.
// Space Complexity: O(M*M)
// Reason: We are using an external array of size ‘M*M’.
int maximumChocolatesSO(vector<vector<int>> &grid)
{
    int n = SZ(grid);
    int m = SZ(grid[0]);
    // Create two 2D vectors 'front' and 'cur' to store computed results
    vector<vector<int>> front(m, vector<int>(m, 0));
    vector<vector<int>> cur(m, vector<int>(m, 0));

    // Initialize 'front' for the last row
    for (int j1 = 0; j1 < m; j1++)
    {
        for (int j2 = 0; j2 < m; j2++)
        {
            if (j1 == j2)
                front[j1][j2] = grid[n - 1][j1];
            else
                front[j1][j2] = grid[n - 1][j1] + grid[n - 1][j2];
        }
    }

    // Outer nested loops for traversing the DP array from the second-to-last row up to the first row
    for (int i = n - 2; i >= 0; i--)
    {
        for (int j1 = 0; j1 < m; j1++)
        {
            for (int j2 = 0; j2 < m; j2++)
            {
                int maxi = INT_MIN;

                // Inner nested loops to try out 9 options (diagonal moves)
                for (int di = -1; di <= 1; di++)
                {
                    for (int dj = -1; dj <= 1; dj++)
                    {
                        int ans;

                        if (j1 == j2)
                            ans = grid[i][j1];
                        else
                            ans = grid[i][j1] + grid[i][j2];

                        // Check if the move is valid (within the grid boundaries)
                        if ((j1 + di < 0 || j1 + di >= m) || (j2 + dj < 0 || j2 + dj >= m))
                            ans += -1e9; // A very large negative value to represent an invalid move
                        else
                            ans += front[j1 + di][j2 + dj]; // Include the value from the 'front' array

                        maxi = max(ans, maxi); // Update the maximum result
                    }
                }
                cur[j1][j2] = maxi; // Store the maximum result for this state in the 'cur' array
            }
        }
        front = cur; // Update 'front' with the values from 'cur' for the next iteration
    }

    // The maximum chocolates that can be collected is stored at the top-left corner of the 'front' array
    return front[0][m - 1];
}

/*##############################DP ON SUBSEQUENCES#################################*/

/*
14. Subset sum equal to target
ANS : Given an array of non-negative integers, and a value sum, determine if there is a subset of the given set with sum equal to given sum.
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC :O(2^n)
// SC :O(ns)
/*
Intuition : We can do this using POWER SEt & RECURSION but here question says if there is one subsets return true thats why i am going to use Recursion
So, we're tying pick and not-pick algo.
Try In the entair array till the index (n-1) does there exist a target that is what the recursion tell you
^Exclude the current element in the subsequence: We first try to find a subsequence without considering the current index element. For this, we will make a recursive call to f(ind-1,target).
^Include the current element in the subsequence: We will try to find a subsequence by considering the current index as element as part of subsequence. As we have included arr[ind], the updated target which we need to find in the rest if the array will be target - arr[ind]. Therefore, we will call f(ind-1,target-arr[ind]).
*/
bool isSubsetSumRecr(int ind, VI &arr, int target)
{
    // Base Case :
    if (target == 0)
        return true; // If we found the target
    if (ind == 0)
        return (arr[ind] == target); // At index 0 if target==arr[0] return ? true :false
    // Pick & not-pick
    bool notPick = isSubsetSumRecr(ind - 1, arr, target);
    bool pick = false;
    if (target >= arr[ind])
    { // target must be smaller than arr elem otherwise how can you sum up or compare same
        pick = isSubsetSumRecr(ind - 1, arr, target - arr[ind]);
    }
    return notPick || pick; // If anyone return true just return true as ans
}
bool isSubsetSumR(vector<int> arr, int sum)
{
    int n = SZ(arr);
    return isSubsetSumRecr(n - 1, arr, sum);
}
// Better ------Memoization----->
// Time Complexity: O(N*K)
// Reason: There are N*K states therefore at max ‘N*K’ new problems will be solved.
// Space Complexity: O(N*K) + O(N)
// Reason: We are using a recursion stack space(O(N)) and a 2D array ( O(N*K)).
bool isSubsetSumMemo(int ind, int target, VI &arr, VVI &dp)
{
    // If the target sum is 0, we have found a subset
    if (target == 0)
        return true;

    // If we have reached the first element in 'arr'
    if (ind == 0)
        return arr[0] == target;

    // If the result for this subproblem has already been computed, return it
    if (dp[ind][target] != -1)
        return dp[ind][target];

    // Try not taking the current element into the subset
    bool notTaken = isSubsetSumMemo(ind - 1, target, arr, dp);

    // Try taking the current element into the subset if it doesn't exceed the target
    bool taken = false;
    if (arr[ind] <= target)
        taken = isSubsetSumMemo(ind - 1, target - arr[ind], arr, dp);

    // Store the result in the dp array to avoid recomputation
    return dp[ind][target] = notTaken || taken;
}

bool isSubsetSumM(vector<int> arr, int sum)
{
    int n = SZ(arr);
    VVI dp(n, VI(sum + 1, -1)); // dp array of size [n][k+1]. The size of the input array is ‘n’, so the index will always lie between ‘0’ and ‘n-1’. The target can take any value between ‘0’ and ‘k’. Therefore we take the dp array as dp[n][k+1]
    return isSubsetSumMemo(n - 1, sum, arr, dp);
}
// Optimal -----Tabulation----->
// Time Complexity: O(N*K)
// Reason: There are two nested loops
// Space Complexity: O(N*K)
// Reason: We are using an external array of size ‘N*K’. Stack Space is eliminated.
bool isSubsetSumT(vector<int> arr, int k)
{
    int n = SZ(arr);
    // Initialize a 2D DP array with dimensions (n x k+1) to store subproblem results
    vector<vector<bool>> dp(n, vector<bool>(k + 1, false));

    // Base case: If the target sum is 0, we can always achieve it by taking no elements
    for (int i = 0; i < n; i++)
    {
        dp[i][0] = true;
    }

    // Base case: If the first element of 'arr' is less than or equal to 'k', set dp[0][arr[0]] to true
    if (arr[0] <= k)
    {
        dp[0][arr[0]] = true;
    }

    // Fill the DP array iteratively
    for (int ind = 1; ind < n; ind++)
    {
        for (int target = 1; target <= k; target++)
        {
            // If we don't take the current element, the result is the same as the previous row
            bool notTaken = dp[ind - 1][target];

            // If we take the current element, subtract its value from the target and check the previous row
            bool taken = false;
            if (arr[ind] <= target)
            {
                taken = dp[ind - 1][target - arr[ind]];
            }

            // Store the result in the DP array for the current subproblem
            dp[ind][target] = notTaken || taken;
        }
    }

    // The final result is stored in dp[n-1][k]
    return dp[n - 1][k];
}
// Most Optimal -----Space Optimization----->
// TC :
// SC :
/*
Intuition :
We see that to calculate a value of a cell of the dp array, we need only the previous row values (say prev). So, we don’t need to store an entire array. Hence we can space optimize it.
Note: Whenever we create a new row ( say cur), we need to explicitly set its first element is true according to our base condition.
*/
bool isSubsetSumSO(vector<int> arr, int k)
{
    int n = SZ(arr);

    // Initialize a vector 'prev' to store the previous row of the DP table
    vector<bool> prev(k + 1, false);

    // Base case: If the target sum is 0, we can always achieve it by taking no elements
    prev[0] = true;

    // Base case: If the first element of 'arr' is less than or equal to 'k', set prev[arr[0]] to true
    if (arr[0] <= k)
    {
        prev[arr[0]] = true;
    }

    // Iterate through the elements of 'arr' and update the DP table
    for (int ind = 1; ind < n; ind++)
    {
        // Initialize a new row 'cur' to store the current state of the DP table
        vector<bool> cur(k + 1, false);

        // Base case: If the target sum is 0, we can achieve it by taking no elements
        cur[0] = true;

        for (int target = 1; target <= k; target++)
        {
            // If we don't take the current element, the result is the same as the previous row
            bool notTaken = prev[target];

            // If we take the current element, subtract its value from the target and check the previous row
            bool taken = false;
            if (arr[ind] <= target)
            {
                taken = prev[target - arr[ind]];
            }

            // Store the result in the current DP table row for the current subproblem
            cur[target] = notTaken || taken;
        }

        // Update 'prev' with the current row 'cur' for the next iteration
        prev = cur;
    }

    // The final result is stored in prev[k]
    return prev[k];
}

/*
15. Partition Equal Subset Sum
ANS : Given an integer array nums, return true if you can partition the array into two subsets such that the sum of the elements in both subsets is equal or false otherwise.
Input :   || Output :
*/
/*
Intuition : If S1==S2==S/2 S=Total Sum
If sum is odd then division is not possible
I'm looking for a subset with sum S/2
>> If you make the total sum of array(6 elems) and its gives you 20 and there are 3elem's sum is 10 then remaiing elems are bound to give you the remaining sum which is S-S/2
So, I need to check if i get one subset with sum of S/2
*/
// Bruteforce ------Recursion----->
// TC :
// SC :
// Better ------Memoization----->
// TC :
// SC :
// Optimal -----Tabulation----->
// TC :
// SC :
// Most Optimal -----Space Optimization----->
// Time Complexity: O(N*K) +O(N)
// Reason: There are two nested loops that account for O(N*K) and at starting we are running a for loop to calculate totSum.
// Space Complexity: O(K)
// Reason: We are using an external array of size ‘K+1’ to store only one row.
bool canPartition(vector<int> &nums)
{
    int totalSum = 0;
    int n = SZ(nums);
    FOR(i, n)
    totalSum += nums[i];
    if (totalSum % 2)
        return false; // If total sum is a odd number then we can't devide into 2 halfs
    int target = totalSum / 2;
    return isSubsetSumSO(nums, target);
}
// Most Most Optimal -----Using Bitset----->
// TC :O(N) Dominated by the iteration through the array
// SC :O(1) For the bitset and sum, ignoring the input array itself
bool canPartitionBit(vector<int> &nums)
{
    int sum = accumulate(nums.begin(), nums.end(), 0);
    if (sum & 1)
        return 0;
    bitset<10000> bits(1);
    for (int i : nums)
        bits |= bits << i;
    return bits[sum >> 1];
}

/*
16. Partition Array Into Two Arrays to Minimize Sum Difference
ANS : You are given an integer array nums of 2 * n integers. You need to partition nums into two arrays of length n to minimize the absolute difference of the sums of the arrays. To partition nums, put each element of nums into one of the two arrays.
Return the minimum possible absolute difference.
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC :
// SC :
// Better -----Memoization------>
// Time Complexity: O(N*totSum) +O(N) +O(N)
// Reason: There are two nested loops that account for O(N*totSum), at starting we are running a for loop to calculate totSum and at last a for loop to traverse the last row.
// Space Complexity: O(N*totSum) + O(N)
// Reason: We are using an external array of size ‘N * totSum’ and a stack space of O(N).
bool subsetSumUtilMemo(int ind, int target, vector<int> &arr, vector<vector<int>> &dp)
{
    // Base case: If the target sum is 0, return true
    if (target == 0)
        return dp[ind][target] = true;

    // Base case: If we have considered all elements and the target is still not 0, return false
    if (ind == 0)
        return dp[ind][target] = (arr[0] == target);

    // If the result for this state is already calculated, return it
    if (dp[ind][target] != -1)
        return dp[ind][target];

    // Recursive cases
    // 1. Exclude the current element
    bool notTaken = subsetSumUtilMemo(ind - 1, target, arr, dp);

    // 2. Include the current element if it doesn't exceed the target
    bool taken = false;
    if (arr[ind] <= target)
        taken = subsetSumUtilMemo(ind - 1, target - arr[ind], arr, dp);

    // Store the result in the DP table and return
    return dp[ind][target] = notTaken || taken;
}
int minSubsetSumDifferenceM(vector<int> &arr, int n)
{
    int totSum = 0;

    // Calculate the total sum of the array
    for (int i = 0; i < n; i++)
    {
        totSum += arr[i];
    }

    // Initialize a DP table to store the results of the subset sum problem
    vector<vector<int>> dp(n, vector<int>(totSum + 1, -1));

    // Calculate the subset sum for each possible sum from 0 to the total sum
    for (int i = 0; i <= totSum; i++)
    {
        bool dummy = subsetSumUtilMemo(n - 1, i, arr, dp);
    }

    int mini = 1e9;
    for (int i = 0; i <= totSum; i++)
    {
        if (dp[n - 1][i] == true)
        {
            int diff = abs(i - (totSum - i));
            mini = min(mini, diff);
        }
    }
    return mini;
}
// Optimal -----Tabulation----->
// Time Complexity: O(N*totSum) +O(N) +O(N)
// Reason: There are two nested loops that account for O(N*totSum), at starting we are running a for loop to calculate totSum, and at last a for loop to traverse the last row.
// Space Complexity: O(N*totSum)
// Reason: We are using an external array of size ‘N * totSum’. Stack Space is eliminated.
int minSubsetSumDifferenceT(vector<int> &arr, int n)
{
    int totSum = 0;

    // Calculate the total sum of the array
    for (int i = 0; i < n; i++)
    {
        totSum += arr[i];
    }

    // Initialize a DP table to store the results of the subset sum problem
    vector<vector<bool>> dp(n, vector<bool>(totSum + 1, false));

    // Base case: If no elements are selected (sum is 0), it's a valid subset
    for (int i = 0; i < n; i++)
    {
        dp[i][0] = true;
    }

    // Initialize the first row based on the first element of the array
    if (arr[0] <= totSum)
        dp[0][totSum] = true;

    // Fill in the DP table using a bottom-up approach
    for (int ind = 1; ind < n; ind++)
    {
        for (int target = 1; target <= totSum; target++)
        {
            // Exclude the current element
            bool notTaken = dp[ind - 1][target];

            // Include the current element if it doesn't exceed the target
            bool taken = false;
            if (arr[ind] <= target)
                taken = dp[ind - 1][target - arr[ind]];

            dp[ind][target] = notTaken || taken;
        }
    }

    int mini = 1e9;
    for (int i = 0; i <= totSum; i++)
    {
        if (dp[n - 1][i] == true)
        {
            // Calculate the absolute difference between two subset sums
            int diff = abs(i - (totSum - i));
            mini = min(mini, diff);
        }
    }
    return mini;
}
// Most Optimal -----Space Optimization----->
// Time Complexity: O(N*totSum) +O(N) +O(N)
// Reason: There are two nested loops that account for O(N*totSum), at starting we are running a for loop to calculate totSum and at last a for loop to traverse the last row.
// Space Complexity: O(totSum)
// Reason: We are using an external array of size ‘totSum+1’ to store only one row.
int minSubsetSumDifferenceSO(vector<int> &arr, int n)
{
    int totSum = 0;

    // Calculate the total sum of the array
    for (int i = 0; i < n; i++)
    {
        totSum += arr[i];
    }

    // Initialize a boolean vector 'prev' to represent the previous row of the DP table
    vector<bool> prev(totSum + 1, false);

    // Base case: If no elements are selected (sum is 0), it's a valid subset
    prev[0] = true;

    // Initialize the first row based on the first element of the array
    if (arr[0] <= totSum)
        prev[arr[0]] = true;

    // Fill in the DP table using a bottom-up approach
    for (int ind = 1; ind < n; ind++)
    {
        // Create a boolean vector 'cur' to represent the current row of the DP table
        vector<bool> cur(totSum + 1, false);
        cur[0] = true;

        for (int target = 1; target <= totSum; target++)
        {
            // Exclude the current element
            bool notTaken = prev[target];

            // Include the current element if it doesn't exceed the target
            bool taken = false;
            if (arr[ind] <= target)
                taken = prev[target - arr[ind]];

            cur[target] = notTaken || taken;
        }

        // Set 'cur' as the 'prev' for the next iteration
        prev = cur;
    }

    int mini = 1e9;
    for (int i = 0; i <= totSum; i++)
    {
        if (prev[i] == true)
        {
            // Calculate the absolute difference between two subset sums
            int diff = abs(i - (totSum - i));
            mini = min(mini, diff);
        }
    }
    return mini;
}

/*
17. Count Subsets with Sum K
ANS : You are given an array 'arr' of size 'n' containing positive integers and a target sum 'k'.
Find the number of ways of selecting the elements from the array such that the sum of chosen elements is equal to the target 'k'.
Since the number of ways can be very large, print it modulo 10 ^ 9 + 7.

Input :   || Output :
*/
// Bruteforce ------Recursion----->
// Time Complexity: O(N*K)
// Reason: There are N*K states therefore at max ‘N*K’ new problems will be solved.
// Space Complexity: O(N)
// Reason: We are using a recursion stack space(O(N))
int findWaysRecr(int ind, int target, VI &arr)
{
    // if (target == 0)
    //     return 1;
    // Or you can add this line
    if (ind == 0)
    {
        if (target == 0 && arr[0] == 0)
            return 2;
        if (target == 0 || target == arr[0])
            return 1;
        return 0;
    }
    // if (ind == 0)
    //     return arr[ind] == target;
    int notPick = findWaysRecr(ind - 1, target, arr);
    int pick = (arr[ind] <= target) ? findWaysRecr(ind - 1, target - arr[ind], arr) : 0;
    return (notPick + pick);
}
int findWaysR(vector<int> &arr, int k)
{
    int n = SZ(arr);
    return findWaysRecr(n - 1, k, arr);
}
// Better -----Memoization------>
// Time Complexity: O(N*K)
// Reason: There are N*K states therefore at max ‘N*K’ new problems will be solved.
// Space Complexity: O(N*K) + O(N)
// Reason: We are using a recursion stack space(O(N)) and a 2D array ( O(N*K)).
int findWaysMemo(int ind, int target, VI &arr, VVI &dp)
{
    // if (target == 0)
    //     return 1;
    // Or you can add this line
    if (ind == 0)
    {
        if (target == 0 && arr[0] == 0)
            return 2;
        if (target == 0 || target == arr[0])
            return 1;
        return 0;
    }
    // if (ind == 0)
    //     return arr[ind] == target;

    if (dp[ind][target] != -1)
        return dp[ind][target];
    int notPick = findWaysMemo(ind - 1, target, arr, dp);
    int pick = 0;
    if (arr[ind] <= target)
        pick = findWaysMemo(ind - 1, target - arr[ind], arr, dp);
    return dp[ind][target] = (notPick + pick);
}
int findWaysM(vector<int> &arr, int k)
{
    int n = SZ(arr);
    VVI dp(n, VI(k + 1, -1));
    return findWaysMemo(n - 1, k, arr, dp);
}
// Optimal -----Tabulation----->
// Time Complexity: O(N*K)
// Reason: There are two nested loops
// Space Complexity: O(N*K)
// Reason: We are using an external array of size ‘N*K’. Stack Space is eliminated.
int findWaysT(vector<int> &arr, int k)
{
    int n = SZ(arr);
    VVI dp(n, VI(k + 1, 0));
    // Base case: If the target sum is 0, there is one valid subset (the empty subset)
    for (int i = 0; i < n; i++)
    {
        dp[i][0] = 1;
    }

    // Initialize the first row based on the first element of the array
    if (arr[0] <= k)
    {
        dp[0][arr[0]] = 1;
    }

    // Fill in the DP table using a bottom-up approach
    for (int ind = 1; ind < n; ind++)
    {
        for (int target = 1; target <= k; target++)
        {
            // Exclude the current element
            int notTaken = dp[ind - 1][target];

            // Include the current element if it doesn't exceed the target
            int taken = 0;
            if (arr[ind] <= target)
            {
                taken = dp[ind - 1][target - arr[ind]];
            }

            // Update the DP table
            dp[ind][target] = notTaken + taken;
        }
    }

    // The final result is in the last cell of the DP table
    return dp[n - 1][k];
}
// Most Optimal -----Space Optimization----->
// Time Complexity: O(N*K)
// Reason: There are two nested loops
// Space Complexity: O(K)
// Reason: We are using an external array of size ‘K+1’ to store only one row.
int findWaysSO(vector<int> &num, int k)
{
    int n = num.size();

    // Initialize a vector 'prev' to represent the previous row of the DP table
    vector<int> prev(k + 1, 0);

    // Base case: If the target sum is 0, there is one valid subset (the empty subset)
    prev[0] = 1;

    // Initialize the first row based on the first element of the array
    if (num[0] <= k)
    {
        prev[num[0]] = 1;
    }

    // Fill in the DP table using a bottom-up approach
    for (int ind = 1; ind < n; ind++)
    {
        // Create a vector 'cur' to represent the current row of the DP table
        vector<int> cur(k + 1, 0);

        cur[0] = 1;

        for (int target = 1; target <= k; target++)
        {
            // Exclude the current element
            int notTaken = prev[target];

            // Include the current element if it doesn't exceed the target
            int taken = 0;
            if (num[ind] <= target)
            {
                taken = prev[target - num[ind]];
            }

            // Update the current row of the DP table
            cur[target] = notTaken + taken;
        }

        // Set 'cur' as 'prev' for the next iteration
        prev = cur;
    }

    // The final result is in the last cell of the 'prev' vector
    return prev[k];
}

/*
18. Count Partitions with Given Difference
ANS : We are given an array ‘ARR’ with N positive integers and an integer D. We need to count the number of ways we can partition the given array into two subsets, S1 and S2 such that S1 - S2 = D and S1 is always greater than or equal to S2.
Input :  arr={5,2,6,4} D=3 || Output :
So, prob states that make partition such that S1>=S2 && S1-S2==D==3
Intuition : As per condition,
S1-S2=D & S1>S2
or, S1=TotSum-S2
or, TotSum-S2-S2=D
or, TotSum - D=2xS2
or, S2=(TotSum-D)/2 : Here qs boils down to find the count of subsets whose sum is (TotSum-D)/2
Edge case are : Constrains are arr[i] will be >=0 so we know from here that TotSum-D >= 0 & has to be even
*/

// Bruteforce ------Recursion----->
// Time Complexity: O(N*K)
// Reason: There are N*K states therefore at max ‘N*K’ new problems will be solved.
// Space Complexity: O(N)
// Reason: We are using a recursion stack space(O(N))

int countPartitionsR(int d, vector<int> &arr)
{
    int n = SZ(arr);
    // RECURSION
    int totSum = 0;
    trav(it, arr) totSum += it;

    // Checking for edge cases
    if (totSum - d < 0 || (totSum - d) % 2)
        return false;

    int s2 = (totSum - d) / 2;
    return findWaysRecr(n - 1, s2, arr);
}
// Better -----Memoization------>
// Time Complexity: O(N*K)
// Reason: There are N*K states therefore at max ‘N*K’ new problems will be solved.
// Space Complexity: O(N*K) + O(N)
// Reason: We are using a recursion stack space(O(N)) and a 2D array ( O(N*K)).

int countPartitionsM(int d, vector<int> &arr)
{
    int n = SZ(arr);
    int totSum = 0;
    trav(it, arr) totSum += it;

    // Checking for edge cases
    if (totSum - d < 0 || (totSum - d) % 2)
        return false;

    int s2 = (totSum - d) / 2;
    VVI dp(n, VI(s2 + 1, -1));
    return findWaysMemo(n - 1, s2, arr, dp);
}
// Optimal -----Tabulation----->
// Time Complexity: O(N*K)
// Reason: There are two nested loops
// Space Complexity: O(N*K)
// Reason: We are using an external array of size ‘N*K’. Stack Space is eliminated.

int findWaysPartyTabu(vector<int> &num, int tar)
{
    int n = num.size();

    vector<vector<int>> dp(n, vector<int>(tar + 1, 0));

    if (num[0] == 0)
        dp[0][0] = 2; // 2 cases -pick and not pick
    else
        dp[0][0] = 1; // 1 case - not pick

    if (num[0] != 0 && num[0] <= tar)
        dp[0][num[0]] = 1; // 1 case -pick

    for (int ind = 1; ind < n; ind++)
    {
        for (int target = 0; target <= tar; target++)
        {

            int notTaken = dp[ind - 1][target];

            int taken = 0;
            if (num[ind] <= target)
                taken = dp[ind - 1][target - num[ind]];

            dp[ind][target] = (notTaken + taken) % mod;
        }
    }
    return dp[n - 1][tar];
}

int countPartitionsT(int d, vector<int> &arr)
{
    int n = SZ(arr);
    int totSum = 0;
    for (int i = 0; i < n; i++)
    {
        totSum += arr[i];
    }

    // Checking for edge cases
    if (totSum - d < 0 || (totSum - d) % 2)
        return 0;

    return findWaysPartyTabu(arr, (totSum - d) / 2);
}
// Most Optimal -----Space Optimization----->
// Time Complexity: O(N*K)
// Reason: There are three nested loops
// Space Complexity: O(K)
// Reason: We are using an external array of size ‘K+1’ to store only one row.
int findWaysPartySO(vector<int> &num, int tar)
{
    int n = num.size();

    vector<int> prev(tar + 1, 0);

    if (num[0] == 0)
        prev[0] = 2; // 2 cases -pick and not pick
    else
        prev[0] = 1; // 1 case - not pick

    if (num[0] != 0 && num[0] <= tar)
        prev[num[0]] = 1; // 1 case -pick

    for (int ind = 1; ind < n; ind++)
    {
        vector<int> cur(tar + 1, 0);
        for (int target = 0; target <= tar; target++)
        {
            int notTaken = prev[target];

            int taken = 0;
            if (num[ind] <= target)
                taken = prev[target - num[ind]];

            cur[target] = (notTaken + taken) % mod;
        }
        prev = cur;
    }
    return prev[tar];
}

int countPartitionsSO(int d, vector<int> &arr)
{
    int n = SZ(arr);
    int totSum = 0;
    for (int i = 0; i < n; i++)
    {
        totSum += arr[i];
    }

    // Checking for edge cases
    if (totSum - d < 0 || (totSum - d) % 2)
        return 0;

    return findWaysPartySO(arr, (totSum - d) / 2);
}

/*
19. 0 1 Knapsack
ANS : A thief is robbing a store and can carry a maximal weight of W into his knapsack. There are N items and the ith item weighs wi and is of value vi. Considering the constraints of the maximum weight that a knapsack can carry, you have to find and return the maximum value that a thief can generate by stealing items.
Input :   || Output :
*/
// #### Its diffrent from the Greedy Algo cz here you can't break the items so thats why its called  0/1 Knapsack
/*
Intuition : Here we can't break items so we're not able to use Greedy Alfo instead we're using Recursion and Pick & notPick method
Base Case :
Boundry not Exceed
Weight is not bigger than maxWeight
*/
// Bruteforce ------Recursion----->
// Time Complexity: O(N*W)
// Reason: There are N*W states therefore at max ‘N*W’ new problems will be solved.
// Space Complexity:  O(N)
// Reason: We are using a recursion stack space(O(N))
int knapsackRecr(vector<int> wt, vector<int> val, int ind, int W)
{
    // Boundry not exceed && weight <=W
    if (ind == 0)
    {
        if (wt[0] <= W)
            return val[0];
        return 0;
    }
    int notPick = knapsackRecr(wt, val, ind - 1, W);
    int pick = (wt[ind] <= W) ? val[ind] + knapsackRecr(wt, val, ind - 1, W - wt[ind]) : INT_MIN;
    return max(pick, notPick);
}
int knapsackR(vector<int> weight, vector<int> value, int maxWeight)
{
    int n = SZ(weight); // any one cz both have same elems
    return knapsackRecr(weight, value, n - 1, maxWeight);
}
// Better -----Memoization------>
// Time Complexity: O(N*W)
// Reason: There are N*W states therefore at max ‘N*W’ new problems will be solved.
// Space Complexity: O(N*W) + O(N)
// Reason: We are using a recursion stack space(O(N)) and a 2D array ( O(N*W)).
int knapsackMemo(vector<int> wt, vector<int> val, int ind, int W, VVI &dp)
{
    // Boundry not exceed && weight <=W
    if (ind == 0)
    {
        if (wt[0] <= W)
            return val[0];
        return 0;
    }
    if (dp[ind][W] != -1)
        return dp[ind][W];
    int notPick = knapsackMemo(wt, val, ind - 1, W, dp);
    int pick = (wt[ind] <= W) ? val[ind] + knapsackMemo(wt, val, ind - 1, W - wt[ind], dp) : INT_MIN;
    return dp[ind][W] = max(pick, notPick);
}
int knapsackM(vector<int> weight, vector<int> value, int maxWeight)
{
    int n = SZ(weight); // any one cz both have same elems
    VVI dp(n, VI(maxWeight + 1, -1));
    return knapsackMemo(weight, value, n - 1, maxWeight, dp);
}
// Optimal -----Tabulation----->
// Time Complexity: O(N*W)
// Reason: There are two nested loops
// Space Complexity: O(N*W)
// Reason: We are using an external array of size ‘N*W’. Stack Space is eliminated.
int knapsackT(vector<int> wt, vector<int> val, int W)
{
    int n = SZ(wt); // any one cz both have same elems
    VVI dp(n, VI(W + 1, 0));
    // Base condition: Fill in the first row for the weight of the first item
    for (int i = wt[0]; i <= W; i++)
    {
        dp[0][i] = val[0];
    }
    // Fill in the DP table using a bottom-up approach
    for (int ind = 1; ind < n; ind++)
    {
        for (int cap = 0; cap <= W; cap++)
        {
            // Calculate the maximum value by either excluding the current item or including it
            int notTaken = dp[ind - 1][cap];
            int taken = INT_MIN;

            // Check if the current item can be included without exceeding the knapsack's capacity
            if (wt[ind] <= cap)
            {
                taken = val[ind] + dp[ind - 1][cap - wt[ind]];
            }

            // Update the DP table
            dp[ind][cap] = max(notTaken, taken);
        }
    }

    // The final result is in the last cell of the DP table
    return dp[n - 1][W];
}
// Most Optimal -----Space Optimization----->
// Time Complexity: O(N*W)
// Reason: There are two nested loops.
// Space Complexity: O(W)
// Reason: We are using an external array of size ‘W+1’ to store only one row.
int knapsackSO(vector<int> &wt, vector<int> &val, int W)
{
    int n = SZ(wt);
    // Initialize a vector 'prev' to represent the previous row of the DP table
    vector<int> prev(W + 1, 0);

    // Base condition: Fill in 'prev' for the weight of the first item
    for (int i = wt[0]; i <= W; i++)
    {
        prev[i] = val[0];
    }

    // Fill in the DP table using a bottom-up approach
    for (int ind = 1; ind < n; ind++)
    {
        for (int cap = W; cap >= 0; cap--)
        {
            // Calculate the maximum value by either excluding the current item or including it
            int notTaken = prev[cap];
            int taken = INT_MIN;

            // Check if the current item can be included without exceeding the knapsack's capacity
            if (wt[ind] <= cap)
            {
                taken = val[ind] + prev[cap - wt[ind]];
            }

            // Update 'prev' for the current capacity
            prev[cap] = max(notTaken, taken);
        }
    }

    // The final result is in the last cell of the 'prev' vector
    return prev[W];
}
/*
20.
ANS :
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC :
// SC :
// Better -----Memoization------>
// TC :
// SC :
// Optimal -----Tabulation----->
// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :
/*
21.
ANS :
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC :
// SC :
// Better -----Memoization------>
// TC :
// SC :
// Optimal -----Tabulation----->
// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :

/*
22.
ANS :
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC :
// SC :
// Better -----Memoization------>
// TC :
// SC :
// Optimal -----Tabulation----->
// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :

/*
23.
ANS :
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC :
// SC :
// Better -----Memoization------>
// TC :
// SC :
// Optimal -----Tabulation----->
// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :

/*
24.
ANS :
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC :
// SC :
// Better -----Memoization------>
// TC :
// SC :
// Optimal -----Tabulation----->
// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :

/*
25.
ANS :
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC :
// SC :
// Better -----Memoization------>
// TC :
// SC :
// Optimal -----Tabulation----->
// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :

/*
26.
ANS :
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC :
// SC :
// Better -----Memoization------>
// TC :
// SC :
// Optimal -----Tabulation----->
// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :

/*
27.
ANS :
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC :
// SC :
// Better -----Memoization------>
// TC :
// SC :
// Optimal -----Tabulation----->
// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :

/*
28.
ANS :
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC :
// SC :
// Better -----Memoization------>
// TC :
// SC :
// Optimal -----Tabulation----->
// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :

/*
29.
ANS :
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC :
// SC :
// Better -----Memoization------>
// TC :
// SC :
// Optimal -----Tabulation----->
// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :

/*
30.
ANS :
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC :
// SC :
// Better -----Memoization------>
// TC :
// SC :
// Optimal -----Tabulation----->
// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :
/*
31.
ANS :
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC :
// SC :
// Better -----Memoization------>
// TC :
// SC :
// Optimal -----Tabulation----->
// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :

/*
32.
ANS :
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC :
// SC :
// Better -----Memoization------>
// TC :
// SC :
// Optimal -----Tabulation----->
// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :

/*
33.
ANS :
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC :
// SC :
// Better -----Memoization------>
// TC :
// SC :
// Optimal -----Tabulation----->
// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :

/*
34.
ANS :
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC :
// SC :
// Better -----Memoization------>
// TC :
// SC :
// Optimal -----Tabulation----->
// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :

/*
35.
ANS :
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC :
// SC :
// Better -----Memoization------>
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
    // // memset(dp,-1,sizeof dp);
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
    // VI h = {2, 7, 9, 3, 2};
    // cout << "Recr " << robRecr(h) << endl;
    // cout << "Memo " << robMemo(h) << endl;
    // cout << "Tab " << robTabu(h) << endl;
    // cout << "S opti " << robSopti(h) << endl;
    // cout << "S opti " << robII(h) << endl;
    // VVI pts = {{1, 2, 5}, {3, 1, 1}, {3, 3, 3}};
    // cout << "Maximum points " << maximumPointsRecr(pts, 3) << endl;
    // cout << "Maximum points " << maximumPointsMemo(pts, 3) << endl;
    // cout << "Maximum points " << maximumPointsTabu(pts, 3) << endl;
    // cout << "Maximum points " << maximumPointsSopti(pts, 3) << endl;
    // cout << "All paths " << allPaths(2, 2) << endl;
    // cout << "All paths " << uniquePathsRecr(2, 2) << endl;
    // cout << "All paths " << uniquePathsMemo(2, 2) << endl;
    // cout << "All paths " << uniquePathsTabu(2, 2) << endl;
    // cout << "All paths " << uniquePathsSopti(2, 2) << endl;
    // VVI obs = {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}};
    // cout << "Recr " << uniquePathsWithObstaclesRecr(obs) << endl;
    // cout << "Memo " << uniquePathsWithObstaclesMemo(obs) << endl;
    // cout << "Tabu " << uniquePathsWithObstaclesTabu(obs) << endl;
    // cout << "Tabu " << uniquePathsWithObstaclesSopti(obs) << endl;
    // VVI grid = {{1, 3, 1}, {1, 5, 1}, {4, 2, 1}};
    // cout << "Min path R " << minPathSum(grid) << endl;
    // cout << "Min path M " << minPathSumMemo(grid) << endl;
    // cout << "Min path T " << minPathSumTabu(grid) << endl;
    // cout << "Min path S " << minPathSumSO(grid) << endl;
    // VVI tri = {{2},
    //            {3, 4},
    //            {6, 5, 7},
    //            {4, 1, 8, 3}};
    // cout << "Min path R " << minimumTotalR(tri) << endl;
    // cout << "Min path M " << minimumTotalM(tri) << endl;
    // cout << "Min path T " << minimumTotalT(tri) << endl;
    // cout << "Min path S " << minimumTotalSO(tri) << endl;
    // VVI tri = {{2, 1, 3},
    //            {6, 5, 4},
    //            {7, 8, 9}};
    // cout << "Min path R " << minFallingPathSumR(tri) << endl;
    // cout << "Min path M " << minFallingPathSumM(tri) << endl;
    // cout << "Min path T " << minFallingPathSumT(tri) << endl;
    // cout << "Min path S " << minFallingPathSumSO(tri) << endl;
    // VVI tri = {{3, 1, 1}, {2, 5, 1}, {1, 5, 5}, {2, 1, 1}};
    // cout << "Max choco R " << maximumChocolatesR(tri) << endl;
    // cout << "Max choco M " << maximumChocolatesM(tri) << endl;
    // cout << "Max choco T " << maximumChocolatesT(tri) << endl;
    // cout << "Max choco S " << maximumChocolatesSO(tri) << endl;

    // VI tri = {2, 3, 3, 3, 4, 5};
    // cout << "Sum R " << isSubsetSumR(tri, 9) << endl;
    // cout << "Sum M " << isSubsetSumM(tri, 9) << endl;
    // cout << "Sum T " << isSubsetSumT(tri, 9) << endl;
    // cout << "Sum S " << isSubsetSumSO(tri, 9) << endl;
    // cout << "Can partition " << canPartition(tri) << endl;
    // cout << "Can partition By Bit " << canPartitionBit(tri) << endl;

    // VI tri = {-36, 36};
    // // cout << "Min subset R " << isSubsetSumR(tri) << endl;
    // cout << "Min subset M " << minSubsetSumDifferenceM(tri, 2) << endl;
    // cout << "Min subset T " << minSubsetSumDifferenceT(tri, 2) << endl;
    // cout << "Min subset S " << minSubsetSumDifferenceSO(tri, 2) << endl;

    // VI tri = {1, 1, 4};
    // cout << "Find ways R " << findWaysR(tri, 5) << endl;
    // cout << "Find ways M " << findWaysM(tri, 5) << endl;
    // cout << "Find ways T " << findWaysT(tri, 5) << endl;
    // cout << "Find ways S " << findWaysSO(tri, 5) << endl;

    // VI tri = {1, 0, 8, 5, 1, 4};
    // cout << "Count Party R " << countPartitionsR(17, tri) << endl;
    // cout << "Count Party M " << countPartitionsM(17, tri) << endl;
    // cout << "Count Party T " << countPartitionsT(17, tri) << endl;
    // cout << "Count Party S " << countPartitionsT(17, tri) << endl;

    vector<int> wt = {1, 2, 4, 5};
    vector<int> val = {5, 4, 8, 6};
    cout << "0 1 Knapsack R " << knapsackR(wt, val, 5) << endl;
    cout << "0 1 Knapsack M " << knapsackM(wt, val, 5) << endl;
    cout << "0 1 Knapsack T " << knapsackT(wt, val, 5) << endl;
    cout << "0 1 Knapsack S " << knapsackSO(wt, val, 5) << endl;

    //  End code here-------->>

    return 0;
}
