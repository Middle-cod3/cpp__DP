#include <bits/stdc++.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <limits>
#include <vector>
using namespace std;
typedef vector<int> VI;
typedef vector<vector<int>> VVI;
typedef vector<vector<vector<int>>> VVVI;
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
#define FORE(i, n) for (int i = 0; i <= n; i++)
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
***************REMEMBER PART************************
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
iii. Try a loop(which are changing params)
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
How to write Base Case ???????
----->>Assume that you're starting a n-1 and you always look at the last index which is 0 and then you start thinking
in terms of a single array containing a single elem and a possible target, Assume that array contain 6 and target is 7, so can you achive this target ?No,so you have to return something.

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

🔟 REMEMBER : Whenever there is a infinite supply of anything, multiple use such statement always when you consider Pick at the same index. It won't stand at a same index bcz we reducing the target

1️⃣1️⃣ Shortcut trick for DP on String *******
i. Express ind1 & ind2 (when it talk about 2 string its mean string 1= string[0...ind1] string 2 = string [0...ind2])
ii. Explore possibilities on that index
iii. Take the best amoung them || Return summation of all possibilities
iv. Write Base Case

1️⃣2️⃣ Why we're doing 0-based to 1-based indexing in String Memoization problems?
-->>As we're checking for negative index in 0-based indexing it will take lill time so we converted it to 1-based and checking upto i==0 so we're not going beyond 0 and its save time lill.
1️⃣3️⃣ Subsequences : Picking any elems from the array with maintainig the order.
Subsets : Picking any elems from the array but no need to maintainig the order.
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
        return dp[n];

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
    return dp[i] = ans;
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
    // Base case :Signifies reaching the starting point, where no cost is incurred because no jumps are needed from the first stone.
    // This represents the minimum cost of 0 when the frog is already at the initial position.
    if (ind == 0)
        return 0;

    int minStep = INT_MAX;
    for (int j = 1; j <= k; j++)
    {
        if (ind - j >= 0) // Here, ind is total stairs and j is no of jumps rthat can someone make
        {
            int jump = minimizeCostRecr(ind - j, k, h) + abs(h[ind] - h[ind - j]);
            minStep = min(minStep, jump);
        }
        else
            break;
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
    // Base case: If we are at the beginning (index 0), no cost is needed.
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
    cout << m << " " << n << endl;
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
    cout << i << " " << j << endl;
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
20. Minimum Coins
ANS : You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.
Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.
You may assume that you have an infinite number of each kind of coin.
Input :   || Output :
Note : Here Greedy algo work in some test cases but it faild in some like if coins={9,6,5,1} amount=11
Greedy will return 3 as [9,1,1] but expected ans is 2 [6,5]
*/
// Bruteforce ------Recursion----->
// Time Complexity: O(N*T)
// Reason: There are N*T states therefore at max ‘N*T’ new problems will be solved.
// Space Complexity:  O(N)
// Reason: We are using a recursion stack space(O(N))
/*
Intuition : Is simple use pick and not pick method but in pick : Now here is the catch, as there is an unlimited supply of coins, we want to again form a solution with the same coin value. So we will not recursively call for f(ind-1, T-arr[ind]) rather we will stay at that index only and call for f(ind, T-arr[ind]) to find the answer.
Note: We will consider the current coin only when its denomination value (arr[ind]) is less than or equal to the target T.
Base case : The base case is triggered when the ind (index) is 0, meaning we are considering only the smallest denomination of coins.
Logic:
Check if the amount is exactly divisible by the coin at index 0 (coins[0]).
If true, return the quotient (amount / coins[0]) as the minimum number of coins needed.
If false, return INT_MAX to indicate that it's not possible to form the amount using only this coin.
Purpose: This ensures that when the recursive function reaches the smallest denomination, it can determine if the remaining amount can be exactly formed using that denomination alone.
*/

/*
REMEMBER : Whenever there is a infinite supply of anything, multiple use such statement always when you consider Pick at the same index. It won't stand at a same index bcz we reducing the target
*/
int coinChangeRecr(int ind, vector<int> &coins, int amount)
{
    // Base case :
    if (ind == 0)
    {
        if (amount % coins[ind] == 0)
            return amount / coins[ind];
        return 1e9;
    }
    int notPick = coinChangeRecr(ind - 1, coins, amount);
    int pick = (coins[ind] <= amount)
                   ? 1 + coinChangeRecr(ind, coins, amount - coins[ind])
                   : INT_MAX;
    return min(notPick, pick);
}
int coinChangeR(vector<int> &coins, int amount)
{
    int n = SZ(coins);
    int result = coinChangeRecr(n - 1, coins, amount);
    return result == 1e9 ? -1 : result;
}
// Better -----Memoization------>
// Time Complexity: O(N*T)
// Reason: There are N*T states therefore at max ‘N*T’ new problems will be solved.
// Space Complexity: O(N*T) + O(N)
// Reason: We are using a recursion stack space(O(N)) and a 2D array ( O(N*T)).
int coinChangeMemo(int ind, vector<int> &coins, int amount, VVI &dp)
{
    // Base case :
    if (ind == 0)
    {
        if (amount % coins[ind] == 0)
            return amount / coins[ind];
        return 1e9;
    }
    if (dp[ind][amount] != -1)
        return dp[ind][amount];
    int notPick = coinChangeMemo(ind - 1, coins, amount, dp);
    int pick = (coins[ind] <= amount)
                   ? 1 + coinChangeMemo(ind, coins, amount - coins[ind], dp)
                   : INT_MAX;
    return dp[ind][amount] = min(notPick, pick);
}
int coinChangeM(vector<int> &coins, int amount)
{
    int n = SZ(coins);
    VVI dp(n, VI(amount + 1, -1));
    int result = coinChangeMemo(n - 1, coins, amount, dp);
    return result == 1e9 ? -1 : result;
}
// Optimal -----Tabulation----->
// Time Complexity: O(N*T)
// Reason: There are two nested loops
// Space Complexity: O(N*T)
// Reason: We are using an external array of size ‘N*T’. Stack Space is eliminated.
int coinChangeT(vector<int> &coins, int amount)
{
    int n = SZ(coins);
    VVI dp(n, VI(amount + 1, 0));
    // Initialize the first row of the DP table
    for (int i = 0; i <= amount; i++)
    {
        if (i % coins[0] == 0)
            dp[0][i] = i / coins[0];
        else
            dp[0][i] = 1e9; // Set it to a very large value if not possible
    }
    // Fill the DP table using a bottom-up approach
    for (int ind = 1; ind < n; ind++)
    {
        for (int target = 0; target <= amount; target++)
        {
            // Calculate the minimum elements needed without taking the current element
            int notTake = dp[ind - 1][target];

            // Calculate the minimum elements needed by taking the current element
            int take = 1e9; // Initialize 'take' to a very large value
            if (coins[ind] <= target)
                take = 1 + dp[ind][target - coins[ind]];

            // Store the minimum of 'notTake' and 'take' in the DP table
            dp[ind][target] = min(notTake, take);
        }
    }

    // The answer is in the bottom-right cell of the DP table
    int ans = dp[n - 1][amount];

    // If 'ans' is still very large, it means it's not possible to form the target sum
    if (ans >= 1e9)
        return -1;

    return ans; // Return the minimum number of elements needed
}
// Most Optimal -----Space Optimization----->
// Time Complexity: O(N*T)
// Reason: There are two nested loops.
// Space Complexity: O(T)
// Reason: We are using two external arrays of size ‘T+1’.
int coinChangeSO(vector<int> &coins, int amount)
{
    int n = SZ(coins);

    // Create two vectors to store the previous and current DP states
    vector<int> prev(amount + 1, 0);
    vector<int> cur(amount + 1, 0);

    // Initialize the first row of the DP table
    for (int i = 0; i <= amount; i++)
    {
        if (i % coins[0] == 0)
            prev[i] = i / coins[0];
        else
            prev[i] = 1e9; // Set it to a very large value if not possible
    }

    // Fill the DP table using a bottom-up approach
    for (int ind = 1; ind < n; ind++)
    {
        for (int target = 0; target <= amount; target++)
        {
            // Calculate the minimum elements needed without taking the current element
            int notTake = prev[target];

            // Calculate the minimum elements needed by taking the current element
            int take = 1e9; // Initialize 'take' to a very large value
            if (coins[ind] <= target)
                take = 1 + cur[target - coins[ind]];

            // Store the minimum of 'notTake' and 'take' in the current DP state
            cur[target] = min(notTake, take);
        }
        // Update the previous DP state with the current state for the next iteration
        prev = cur;
    }

    // The answer is in the last row of the DP table
    int ans = prev[amount];

    // If 'ans' is still very large, it means it's not possible to form the target sum
    if (ans >= 1e9)
        return -1;

    return ans; // Return the minimum number of elements needed
}
/*
21.Target Sum
ANS : You are given an integer array nums and an integer target.
You want to build an expression out of nums by adding one of the symbols '+' and '-' before each integer in nums and then concatenate all the integers.
For example, if nums = [2, 1], you can add a '+' before 2 and a '-' before 1 and concatenate them to build the expression "+2-1".
Return the number of different expressions that you can build, which evaluates to target.
Input :   || Output :
*/
/*
Intuition : This prob is similar to divide the array into two partitions or subsets such that s1 is sum(subset1) & s2 is sum(subset2) and the diffrenece
is the target that you're looking for.
*/
// Bruteforce ------Recursion----->
// Time Complexity: O(N*K)
// Reason: There are N*K states therefore at max ‘N*K’ new problems will be solved.
// Space Complexity: O(N)
// Reason: We are using a recursion stack space(O(N))
int findTargetSumWaysR(vector<int> &nums, int target)
{
    int n = SZ(nums);
    int totSum = 0;
    trav(it, nums) totSum += it;

    // Checking for edge cases
    if (totSum - target < 0 || (totSum - target) % 2)
        return false;

    int s2 = (totSum - target) / 2;
    return findWaysRecr(n - 1, s2, nums);
}
// Better -----Memoization------>
// Time Complexity: O(N*K)
// Reason: There are N*K states therefore at max ‘N*K’ new problems will be solved.
// Space Complexity: O(N*K) + O(N)
// Reason: We are using a recursion stack space(O(N)) and a 2D array ( O(N*K)).
int findTargetSumWaysM(vector<int> &nums, int target)
{
    return findWaysM(nums, target);
}
// Optimal -----Tabulation----->
// Time Complexity: O(N*K)
// Reason: There are two nested loops
// Space Complexity: O(N*K)
// Reason: We are using an external array of size ‘N*K’. Stack Space is eliminated.
int findTargetSumWaysT(vector<int> &nums, int target)
{
    return findWaysT(nums, target);
}
// Most Optimal -----Space Optimization----->
// Time Complexity: O(N*K)
// Reason: There are two nested loops
// Space Complexity: O(K)
// Reason: We are using an external array of size ‘K+1’ to store only one row.
int findTargetSumWaysSO(vector<int> &nums, int target)
{
    return findWaysSO(nums, target);
}

/*
22. Coin Change II
ANS : You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.
Return the number of combinations that make up that amount. If that amount of money cannot be made up by any combination of the coins, return 0.
You may assume that you have an infinite number of each kind of coin.
The answer is guaranteed to fit into a signed 32-bit integer.

Input :   || Output :
*/
// Bruteforce ------Recursion----->
// Time Complexity: O(N*T)
// Reason: There are N*T states therefore at max ‘N*T’ new problems will be solved.
// Space Complexity:  O(N)
// Reason: We are using a recursion stack space(O(N))
int coinChangeRecrII(int ind, vector<int> &coins, int amount)
{
    // Base case :
    if (amount == 0)
        return 1; // one valid combination found
    if (ind < 0 || amount < 0)
        return 0; // no valid combination

    // Don't pick the current coin
    int notPick = coinChangeRecrII(ind - 1, coins, amount);

    // Pick the current coin
    int pick = (coins[ind] <= amount) ? coinChangeRecrII(ind, coins, amount - coins[ind]) : 0;

    // Return the sum of both choices
    return notPick + pick;
}
int coinChangeRII(vector<int> &coins, int amount)
{
    int n = SZ(coins);
    return coinChangeRecrII(n - 1, coins, amount);
}
// Better -----Memoization------>
// Time Complexity: O(N*T)
// Reason: There are N*T states therefore at max ‘N*T’ new problems will be solved.
// Space Complexity: O(N*T) + O(N)
// Reason: We are using a recursion stack space(O(N)) and a 2D array ( O(N*T)).
int coinChangeMemoII(int ind, VI &coins, int amount, VVI &dp)
{
    // Base case: if we're at the first element
    if (ind == 0)
    {
        // Check if the target sum is divisible by the first element
        return (amount % coins[0] == 0);
    }

    // If the result for this index and target sum is already calculated, return it
    if (dp[ind][amount] != -1)
        return dp[ind][amount];

    // Calculate the number of ways without taking the current element
    int notTaken = coinChangeMemoII(ind - 1, coins, amount, dp);

    // Calculate the number of ways by taking the current element
    int taken = 0;
    if (coins[ind] <= amount)
        taken = coinChangeMemoII(ind, coins, amount - coins[ind], dp);

    // Store the sum of ways in the DP table and return it
    return dp[ind][amount] = notTaken + taken;
}
int coinChangeMII(vector<int> &coins, int amount)
{
    int n = SZ(coins);
    VVI dp(n, VI(amount + 1, -1));
    return coinChangeMemoII(n - 1, coins, amount, dp);
}
// Optimal -----Tabulation----->
// Time Complexity: O(N*T)
// Reason: There are two nested loops
// Space Complexity: O(N*T)
// Reason: We are using an external array of size ‘N*T’. Stack Space is eliminated.
int coinChangeTII(vector<int> &arr, int T)
{
    int n = SZ(arr);
    VVI dp(n, VI(T + 1, 0));

    // Initializing base condition
    for (int i = 0; i <= T; i++)
    {
        if (i % arr[0] == 0)
            dp[0][i] = 1;
        // Else condition is automatically fulfilled,
        // as dp array is initialized to zero
    }

    for (int ind = 1; ind < n; ind++)
    {
        for (int target = 0; target <= T; target++)
        {
            int notTaken = dp[ind - 1][target];

            int taken = 0;
            if (arr[ind] <= target)
                taken = dp[ind][target - arr[ind]];

            dp[ind][target] = notTaken + taken;
        }
    }

    return dp[n - 1][T];
}
// Most Optimal -----Space Optimization----->
// Time Complexity: O(N*T)
// Reason: There are two nested loops.
// Space Complexity: O(T)
// Reason: We are using two external arrays of size ‘T+1’.
int coinChangeSOII(vector<int> &arr, int T)
{
    int n = SZ(arr);

    vector<int> prev(T + 1, 0); // Create a vector to store the previous DP state

    // Initialize base condition
    for (int i = 0; i <= T; i++)
    {
        if (i % arr[0] == 0)
            prev[i] = 1; // There is one way to make change for multiples of the first coin
        // Else condition is automatically fulfilled,
        // as the prev vector is initialized to zero
    }

    for (int ind = 1; ind < n; ind++)
    {
        vector<int> cur(T + 1, 0); // Create a vector to store the current DP state
        for (int target = 0; target <= T; target++)
        {
            int notTaken = prev[target]; // Number of ways without taking the current coin

            int taken = 0;
            if (arr[ind] <= target)
                taken = cur[target - arr[ind]]; // Number of ways by taking the current coin

            cur[target] = notTaken + taken; // Total number of ways for the current target
        }
        prev = cur; // Update the previous DP state with the current state for the next coin
    }

    return prev[T]; // Return the total number of ways to make change for the target
}
/*
23. Unbounded Knapsack || Knapsack with Duplicate Items
ANS : Given a set of N items, each with a weight and a value, represented by the array w and val respectively. Also, a knapsack with weight limit W.
The task is to fill the knapsack in such a way that we can get the maximum profit. Return the maximum profit.

Note: Each item can be taken any number of times.
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// Time Complexity: O(N*W)
// Reason: There are N*W states therefore at max ‘N*W’ new problems will be solved.
// Space Complexity: O(N)
// Reason: We are using a recursion stack space(O(N))
int knapSackRecrII(int ind, int W, int val[], int wt[])
{
    // Base case: if we're at the first item
    if (ind == 0)
    {
        // Calculate and return the maximum value for the given weight limit
        return (W / wt[0]) * val[0];
    }
    int notPick = knapSackRecrII(ind - 1, W, val, wt);
    int pick = (wt[ind] <= W) ? val[ind] + knapSackRecrII(ind, W - wt[ind], val, wt) : INT_MIN;
    return max(notPick, pick);
}
int knapSackRII(int N, int W, int val[], int wt[])
{
    return knapSackRecrII(N - 1, W, val, wt);
}
// Better -----Memoization------>
// Time Complexity: O(N*W)
// Reason: There are N*W states therefore at max ‘N*W’ new problems will be solved.
// Space Complexity: O(N*W) + O(N)
// Reason: We are using a recursion stack space(O(N)) and a 2D array ( O(N*W)).
int knapSackMemoII(int ind, int W, int val[], int wt[], VVI &dp)
{
    // Base case :
    if (ind == 0)
        return (W / wt[0]) * val[0];
    if (dp[ind][W] != -1)
        return dp[ind][W];
    int notPick = knapSackMemoII(ind - 1, W, val, wt, dp);
    int pick = (wt[ind] <= W) ? val[ind] + knapSackMemoII(ind, W - wt[ind], val, wt, dp) : INT_MIN;
    return dp[ind][W] = max(notPick, pick);
}
int knapSackMII(int N, int W, int val[], int wt[])
{
    VVI dp(N, VI(W + 1, -1));
    return knapSackMemoII(N - 1, W, val, wt, dp);
}
// Optimal -----Tabulation----->
// Time Complexity: O(N*W)
// Reason: There are two nested loops
// Space Complexity: O(N*W)
// Reason: We are using an external array of size ‘N*W’. Stack Space is eliminated.
int knapSackTII(int N, int W, int val[], int wt[])
{
    VVI dp(N, VI(W + 1, 0));
    // Base Condition
    for (int i = wt[0]; i <= W; i++)
    {
        dp[0][i] = (i / wt[0]) * val[0]; // Calculate the maximum value for the first item
    }

    for (int ind = 1; ind < N; ind++)
    {
        for (int cap = 0; cap <= W; cap++)
        {
            int notTaken = 0 + dp[ind - 1][cap]; // Maximum value without taking the current item

            int taken = INT_MIN;
            if (wt[ind] <= cap)
                taken = val[ind] + dp[ind][cap - wt[ind]]; // Maximum value by taking the current item

            dp[ind][cap] = max(notTaken, taken); // Store the maximum value in the DP table
        }
    }

    return dp[N - 1][W]; // Return the maximum value considering all items and the knapsack capacity
}
// Most Optimal -----Space Optimization----->
// Time Complexity: O(N*W)
// Reason: There are two nested loops.
// Space Complexity: O(W)
// Reason: We are using an external array of size ‘W+1’ to store only one row.
int knapSackSOII(int N, int W, int val[], int wt[])
{
    vector<int> cur(W + 1, 0); // Create a vector to store the current DP state

    // Base Condition
    for (int i = wt[0]; i <= W; i++)
    {
        cur[i] = (i / wt[0]) * val[0]; // Calculate the maximum value for the first item
    }

    for (int ind = 1; ind < N; ind++)
    {
        for (int cap = 0; cap <= W; cap++)
        {
            int notTaken = cur[cap]; // Maximum value without taking the current item

            int taken = INT_MIN;
            if (wt[ind] <= cap)
                taken = val[ind] + cur[cap - wt[ind]]; // Maximum value by taking the current item

            cur[cap] = max(notTaken, taken); // Store the maximum value in the current DP state
        }
    }

    return cur[W]; // Return the maximum value considering all items and the knapsack capacity
}

/*
24. Rod Cutting
ANS : Given a rod of length N inches and an array of prices, price[]. price[i] denotes the value of a piece of length i. Determine the maximum value obtainable by cutting up the rod and selling the pieces.
Note: Consider 1-based indexing.
Input :   || Output :
Intuition : You can chnage the question as How do you collect rod lengths to make N and while collecting rod lengths maximize the price .
As you can see now its a similer prob as Knapsack prob.
Now, try to pick lengths and sum it up to make N
*/
// Bruteforce ------Recursion----->
// Time Complexity: O(N*W) or exponential its >2^n bcz we're staying at same index
// Reason: There are N*W states therefore at max ‘N*W’ new problems will be solved.
// Space Complexity: O(N*W) + O(N)
// Reason: We are using a recursion stack space(O(N)) and a 2D array ( O(N*W)).
int curRodRecr(int ind, int N, VI &price)
{
    if (ind == 0)
        return N * price[0];
    int notPick = curRodRecr(ind - 1, N, price);
    int rodLen = ind + 1;
    int pick = (rodLen <= N) ? price[ind] + curRodRecr(ind, N - rodLen, price) : INT_MIN;
    return max(notPick, pick);
}
int cutRodR(vector<int> &price, int n)
{
    return curRodRecr(n - 1, n, price);
}
// Better -----Memoization------>
// Time Complexity: O(N*W)
// Reason: There are N*W states therefore at max ‘N*W’ new problems will be solved.
// Space Complexity: O(N*W) + O(N)
// Reason: We are using a recursion stack space(O(N)) and a 2D array ( O(N*W)).
int curRodMemo(int ind, int N, VI &price, VVI &dp)
{
    if (ind == 0)
        return N * price[0];
    if (dp[ind][N] != -1)
        return dp[ind][N];
    int notPick = curRodMemo(ind - 1, N, price, dp);
    int rodLen = ind + 1;
    int pick = (rodLen <= N) ? price[ind] + curRodMemo(ind, N - rodLen, price, dp) : INT_MIN;
    return dp[ind][N] = max(notPick, pick);
}
int cutRodM(vector<int> &price, int n)
{
    VVI dp(n, VI(n + 1, -1));
    return curRodMemo(n - 1, n, price, dp);
}
// Optimal -----Tabulation----->
// Time Complexity: O(N*W)
// Reason: There are two nested loops
// Space Complexity: O(N*W)
// Reason: We are using an external array of size ‘N*W’. Stack Space is eliminated.
int cutRodT(vector<int> &price, int n)
{
    VVI dp(n, VI(n + 1, 0));
    // Base case :
    for (int N = 0; N <= n; N++)
    {
        dp[0][N] = N * price[0];
    }

    for (int ind = 1; ind < n; ind++)
    {
        for (int N = 0; N <= n; N++)
        {
            int notPick = dp[ind - 1][N];
            int rodLen = ind + 1;
            int pick = (rodLen <= N) ? price[ind] + dp[ind][N - rodLen] : INT_MIN;
            dp[ind][N] = max(pick, notPick);
        }
    }
    return dp[n - 1][n];
}
// Most Optimal -----Space Optimization----->
// Time Complexity: O(N*W)
// Reason: There are two nested loops.
// Space Complexity: O(W)
// Reason: We are using an external array of size ‘W+1’ to store only one row.
int cutRodSO(vector<int> &price, int n)
{
    VI prev(n + 1, 0);
    for (int N = 0; N <= n; N++)
    {
        prev[N] = N * price[0];
    }

    for (int ind = 1; ind < n; ind++)
    {
        for (int N = 0; N <= n; N++)
        {
            int notPick = prev[N];
            int rodLen = ind + 1;
            int pick = (rodLen <= N) ? price[ind] + prev[N - rodLen] : INT_MIN;
            prev[N] = max(pick, notPick);
        }
    }
    return prev[n];
}

/*##############################DP ON STRINGS#################################*/

/*
25. Longest Common Subsequence
ANS : Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.

A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

For example, "ace" is a subsequence of "abcde".
A common subsequence of two strings is a subsequence that is common to both strings.
## Bruteforce should be generate all subsequences and then compare for both string but for this Time comp will be exponential in nature
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// Time Complexity: O(N*M)
// Reason: There are N*M states therefore at max ‘N*M’ new problems will be solved.
// Space Complexity: O(N*M)
// Reason: We are using an auxiliary recursion stack space(O(N+M)) (see the recursive tree, in the worst case, we will go till N+M calls at a time)
/*
Intuition : So, we're trying to generate all subsequences & compare on way for generate all subsequences we're twirk some thing.
So, If both string subsequence is matching then increment by one and go for next subsequences for both string.
If its not matched then go for both strings next subsequence individually and return max of this 2 subsequence.
And the base case is : if any moment index is at end of the string means ind1<0 || ind2<0 then the longest length is 0
*/
int longestCommonSubsequenceRecr(int ind1, int ind2, string text1, string text2)
{
    // Base case :
    if (ind1 < 0 || ind2 < 0)
        return 0;

    // If subsequences matched
    if (text1[ind1] == text2[ind2])
        return 1 + longestCommonSubsequenceRecr(ind1 - 1, ind2 - 1, text1, text2);
    int t1 = longestCommonSubsequenceRecr(ind1 - 1, ind2, text1, text2);
    int t2 = longestCommonSubsequenceRecr(ind1, ind2 - 1, text1, text2);
    return max(t1, t2);
}
int longestCommonSubsequenceR(string text1, string text2)
{
    int n = SZ(text1);
    int m = SZ(text2);
    return longestCommonSubsequenceRecr(n - 1, m - 1, text1, text2);
}
// Better -----Memoization------>
// TC :
// SC :
int longestCommonSubsequenceMemo(int ind1, int ind2, string text1, string text2, VVI &dp)
{
    // Base case :
    if (ind1 < 0 || ind2 < 0)
        return 0;
    if (dp[ind1][ind2] != -1)
        return dp[ind1][ind2];
    // If subsequences matched
    if (text1[ind1] == text2[ind2])
        return dp[ind1][ind2] = 1 + longestCommonSubsequenceMemo(ind1 - 1, ind2 - 1, text1, text2, dp);
    int t1 = longestCommonSubsequenceMemo(ind1 - 1, ind2, text1, text2, dp);
    int t2 = longestCommonSubsequenceMemo(ind1, ind2 - 1, text1, text2, dp);
    return dp[ind1][ind2] = max(t1, t2);
}
int longestCommonSubsequenceM(string text1, string text2)
{
    int n = SZ(text1);
    int m = SZ(text2);
    VVI dp(n, VI(m + 1, -1));
    return longestCommonSubsequenceMemo(n - 1, m - 1, text1, text2, dp);
}
// Optimal -----Tabulation----->
// Time Complexity: O(N*M)
// Reason: There are two nested loops
// Space Complexity: O(N*M)
// Reason: We are using an external array of size ‘N*M)’. Stack Space is eliminated.
int longestCommonSubsequenceT(string s1, string s2)
{
    int n = s1.size();
    int m = s2.size();

    vector<vector<int>> dp(n + 1, vector<int>(m + 1, -1)); // Create a DP table

    // Initialize the base cases
    for (int i = 0; i <= n; i++)
    {
        dp[i][0] = 0;
    }
    for (int i = 0; i <= m; i++)
    {
        dp[0][i] = 0;
    }

    // Fill in the DP table to calculate the length of LCS
    for (int ind1 = 1; ind1 <= n; ind1++)
    {
        for (int ind2 = 1; ind2 <= m; ind2++)
        {
            if (s1[ind1 - 1] == s2[ind2 - 1])
                dp[ind1][ind2] = 1 + dp[ind1 - 1][ind2 - 1]; // Characters match, increment LCS length
            else
                dp[ind1][ind2] = max(dp[ind1 - 1][ind2], dp[ind1][ind2 - 1]); // Characters don't match, consider the maximum from left or above
        }
    }

    return dp[n][m]; // Return the length of the Longest Common Subsequence
}
// Most Optimal -----Space Optimization----->
// Time Complexity: O(N*M)
// Reason: There are two nested loops.
// Space Complexity: O(M)
// Reason: We are using an external array of size ‘M+1’ to store only two rows.
int longestCommonSubsequenceSO(string s1, string s2)
{
    int n = s1.size();
    int m = s2.size();

    // Initialize two vectors to store the current and previous rows of the DP table
    vector<int> prev(m + 1, 0), cur(m + 1, 0);

    // Base case is covered as we have initialized the prev and cur vectors to 0.

    for (int ind1 = 1; ind1 <= n; ind1++)
    {
        for (int ind2 = 1; ind2 <= m; ind2++)
        {
            if (s1[ind1 - 1] == s2[ind2 - 1])
                cur[ind2] = 1 + prev[ind2 - 1]; // Characters match, increment LCS length
            else
                cur[ind2] = max(prev[ind2], cur[ind2 - 1]); // Characters don't match, consider the maximum from above or left
        }
        prev = cur; // Update the previous row with the current row
    }

    return prev[m]; // Return the length of the Longest Common Subsequence
}

/*
26. Print Longest Common Subsequence
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
// Time Complexity: O(N*M)
// Reason: There are two nested loops
// Space Complexity: O(N*M)
// Reason: We are using an external array of size ‘N*M’. Stack Space is eliminated.
void lcs(string s1, string s2)
{

    int n = s1.size();
    int m = s2.size();

    vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
    for (int i = 0; i <= n; i++)
    {
        dp[i][0] = 0;
    }
    for (int i = 0; i <= m; i++)
    {
        dp[0][i] = 0;
    }

    for (int ind1 = 1; ind1 <= n; ind1++)
    {
        for (int ind2 = 1; ind2 <= m; ind2++)
        {
            if (s1[ind1 - 1] == s2[ind2 - 1])
                dp[ind1][ind2] = 1 + dp[ind1 - 1][ind2 - 1];
            else
                dp[ind1][ind2] = 0 + max(dp[ind1 - 1][ind2], dp[ind1][ind2 - 1]);
        }
    }

    int len = dp[n][m];
    int i = n;
    int j = m;

    int index = len - 1;
    string str = "";
    for (int k = 1; k <= len; k++)
    {
        str += "$"; // dummy string
    }

    while (i > 0 && j > 0)
    {
        if (s1[i - 1] == s2[j - 1])
        {
            str[index] = s1[i - 1];
            index--;
            i--;
            j--;
        }
        else if (s1[i - 1] > s2[j - 1])
        {
            i--;
        }
        else
            j--;
    }
    cout << str;
}
// Most Optimal -----Space Optimization----->
// TC :
// SC :

/*
27. Longest Common Substring
ANS : Given two strings. The task is to find the length of the longest common substring.

Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC :
// SC :
// Better -----Memoization------>
// TC :
// SC :
// Optimal -----Tabulation----->
// Time Complexity: O(N*M)
// Reason: There are two nested loops
// Space Complexity: O(N*M)
// Reason: We are using an external array of size ‘N*M)’. Stack Space is eliminated.
/*
Intuition : We cant use subsequences method bcz there we're not caring consecutiveness but for you've to so we  need to chnage some conditions.
While finding the longest common subsequence, we were using two pointers (ind1 and ind2) to map the characters of the two strings. We will again have the same set of conditions for finding the longest common substring, with slight modifications to what we do when the condition becomes true.
We will try to form a solution in the bottom-up (tabulation) approach. We will set two nested loops with loop variables i and j.
Thinking in terms of consecutiveness of characters
We have two conditions:

if(S1[i-1] != S2[j-1]), the characters don’t match, therefore the consecutiveness of characters is broken. So we set the cell value (dp[i][j]) as 0.
if(S1[i-1] == S2[j-1]), then the characters match and we simply set its value to 1+dp[i-1][j-1]. We have done so because dp[i-1][j-1] gives us the longest common substring till the last cell character (current strings -{matching character}). As the current cell’s character is matching we are adding 1 to the consecutive chain.
Note: dp[n][m] will not give us the answer; rather the maximum value in the entire dp array will give us the length of the longest common substring. This is because there is no restriction that the longest common substring is present at the end of both the strings.
*/
int longestCommonSubstringT(string s1, string s2)
{
    int n = SZ(s1);
    int m = SZ(s2);

    VVI dp(n + 1, VI(m + 1, -1)); // Create a DP table

    // Initialize the base cases
    FORE(i, n)
    dp[i][0] = 0;
    FORE(i, m)
    dp[0][i] = 0;
    int ans = 0;
    // Fill in the DP table to calculate the length of LCS
    FOR1(ind1, n)
    {
        FOR1(ind2, m)
        {
            if (s1[ind1 - 1] == s2[ind2 - 1])
            {
                dp[ind1][ind2] = 1 + dp[ind1 - 1][ind2 - 1]; // Characters match, increment LCS length
                ans = max(ans, dp[ind1][ind2]);
            }
            else
                dp[ind1][ind2] = 0; // Characters don't match, substring length becomes 0
        }
    }

    return ans;
}
// Most Optimal -----Space Optimization----->
// Time Complexity: O(N*M)
// Reason: There are two nested loops.
// Space Complexity: O(M)
// Reason: We are using an external array of size ‘M+1’ to store only two rows.
int longestCommonSubstringSO(string s1, string s2)
{
    int n = SZ(s1);
    int m = SZ(s2);

    // Initialize two vectors to store the current and previous rows of the DP table
    vector<int> prev(m + 1, 0), cur(m + 1, 0);
    int ans = 0;

    // Base case is covered as we have initialized the prev and cur vectors to 0.

    FORE(ind1, n)
    {
        FORE(ind2, m)
        {
            if (s1[ind1 - 1] == s2[ind2 - 1])
            {
                cur[ind2] = 1 + prev[ind2 - 1]; // Characters match, increment LCS length
                ans = max(ans, cur[ind2 - 1]);
            }
            else
                cur[ind2] = max(prev[ind2], cur[ind2 - 1]); // Characters don't match, consider the maximum from above or left
        }
        prev = cur; // Update the previous row with the current row
    }

    return ans;
}

/*
28.  Longest Palindromic Subsequence
ANS : Given a string s, find the longest palindromic subsequence's length in s.
A subsequence is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.
Input :   || Output :
*/
// #Bruteforce should be Generate all the subsequences then check for palindrome and pick the longest but its TC is going to exponential in nature so we're trying recursion but with some twirke

// Bruteforce ------Recursion----->
// TC :
// SC :
/*
Intuition : As its check for palindrome we can observe from the string is if we reverse the string then we got 2 string and if we do LONGEST COMMON SUBSEQUENCES then we can do both palindrome and subsequence in a single logic
*/
// Better -----Memoization------>
// TC :
// SC :
// Optimal -----Tabulation----->
// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC : O(NxM)
// SC :O(N+M)
int longestPalindromeSubseqSO(string s)
{
    string t = s;
    reverse(t.begin(), t.end());
    return longestCommonSubsequenceSO(s, t);
}

/*
29. Minimum Insertion Steps to Make a String Palindrome
ANS : Given a string s. In one step you can insert any character at any index of the string.
Return the minimum number of steps to make s palindrome.
A Palindrome String is one that reads the same backward as well as forward.
Input :   || Output :
*/
/*
Intuition:
We need to find the minimum insertions required to make a string palindrome.
Let us keep the “minimum” criteria aside and think, how can we make any given string palindrome by inserting characters?
The easiest way is to add the reverse of the string at the back of the original string as shown below. This will make any string palindrome.
Here the number of characters inserted will be equal to n (length of the string). This is the maximum number of characters we can insert to make strings palindrome.
The problem states us to find the minimum of insertions. Let us try to figure it out:
To minimize the insertions, we will first try to refrain from adding those characters again which are already making the given string palindrome. For the given example, “aaa”, “aba”,”aca”, any of these are themselves palindromic components of the string. We can take any of them( as all are of equal length) and keep them intact. (let’s say “aaa”).
Now, there are two characters(‘b’ and ‘c’) remaining which prevent the string from being a palindrome. We can reverse their order and add them to the string to make the entire string palindrome.
We can do this by taking some other components (like “aca”) as well.
In order to minimize the insertions, we need to find the length of the longest palindromic component or in other words, the longest palindromic subsequence.
Minimum Insertion required = n(length of the string) - length of longest palindromic subsequence.
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
// TC : O(Nxm)
// SC :O(M)
int minInsertionSO(string s)
{
    int n = SZ(s);
    int k = longestPalindromeSubseqSO(s);

    // The minimum insertions required is the difference between the string length and its longest palindromic subsequence length
    return n - k;
}

/*
30. Minimum Insertions/Deletions to Convert String
ANS : Minimum Insertions/Deletions to Convert String A to String B

We are given two strings, str1 and str2. We are allowed the following operations:

Delete any number of characters from string str1.
Insert any number of characters in string str1.
We need to tell the minimum operations required to convert str1 to str2.
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
// TC : O(Nxm)
// SC :O(M)
int canYouMakeSO(string str1, string str2)
{
    int n = SZ(str1);
    int m = SZ(str2);

    // Calculate the length of the longest common subsequence between str1 and str2
    int k = longestCommonSubsequenceSO(str1, str2);

    // Calculate the minimum operations required to convert str1 to str2
    return (n - k) + (m - k);
}
/*
31. Shortest Common Supersequence
ANS : Given two strings str1 and str2, return the shortest string that has both str1 and str2 as subsequences. If there are multiple valid strings, return any of them.
A string s is a subsequence of string t if deleting some number of characters from t (possibly 0) results in the string s.
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
// TC : O(N*M)
// SC : O(N*M)
string shortestSupersequence(string s1, string s2)
{

    int n = s1.size();
    int m = s2.size();

    vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
    for (int i = 0; i <= n; i++)
    {
        dp[i][0] = 0;
    }
    for (int i = 0; i <= m; i++)
    {
        dp[0][i] = 0;
    }

    for (int ind1 = 1; ind1 <= n; ind1++)
    {
        for (int ind2 = 1; ind2 <= m; ind2++)
        {
            if (s1[ind1 - 1] == s2[ind2 - 1])
                dp[ind1][ind2] = 1 + dp[ind1 - 1][ind2 - 1];
            else
                dp[ind1][ind2] = 0 + max(dp[ind1 - 1][ind2], dp[ind1][ind2 - 1]);
        }
    }

    int len = dp[n][m];
    int i = n;
    int j = m;

    int index = len - 1;
    string ans = "";

    while (i > 0 && j > 0)
    {
        if (s1[i - 1] == s2[j - 1])
        {
            ans += s1[i - 1];
            index--;
            i--;
            j--;
        }
        else if (dp[i - 1][j] > dp[i][j - 1])
        {
            ans += s1[i - 1];
            i--;
        }
        else
        {
            ans += s2[j - 1];
            j--;
        }
    }

    // Adding Remaing Characters - Only one of the below two while loops will run

    while (i > 0)
    {
        ans += s1[i - 1];
        i--;
    }
    while (j > 0)
    {
        ans += s2[j - 1];
        j--;
    }

    reverse(ans.begin(), ans.end());

    return ans;
}

/*
32. Distinct Subsequences
ANS : Given two strings s and t, return the number of distinct subsequences of s which equals t.
The test cases are generated so that the answer fits on a 32-bit signed integer.
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC : Expponential
// SC : O(n+m)
int numDistinctRecr(string s1, string s2, int ind1, int ind2)
{
    // If s2 has been completely matched, return 1 (found a valid subsequence)
    if (ind2 < 0)
        return 1;

    // If s1 has been completely traversed but s2 hasn't, return 0
    if (ind1 < 0)
        return 0;

    int result = 0;

    // If the characters match, consider two options: either leave one character in s1 and s2
    // or leave one character in s1 and continue matching s2
    if (s1[ind1] == s2[ind2])
    {
        int leaveOne = numDistinctRecr(s1, s2, ind1 - 1, ind2 - 1);
        int stay = numDistinctRecr(s1, s2, ind1 - 1, ind2);

        result = (leaveOne + stay) % mod;
    }
    else
    {
        // If characters don't match, just leave one character in s1 and continue matching s2
        result = numDistinctRecr(s1, s2, ind1 - 1, ind2);
    }

    return result;
}
int numDistinctR(string s, string t)
{
    // Recursion
    int n = SZ(s);
    int m = SZ(t);
    return numDistinctRecr(s, t, n - 1, m - 1);
}
// Better -----Memoization------>
// Time Complexity: O(N*M)
// Reason: There are N*M states therefore at max ‘N*M’ new problems will be solved.
// Space Complexity: O(N*M) + O(N+M)
// Reason: We are using a recursion stack space(O(N+M)) and a 2D array ( O(N*M)).
int numDistinctMemo(string s, string t, int i, int j, VVI &dp)
{
    if (j == 0)
        return 1;
    if (i == 0)
        return 0;
    if (dp[i][j] != -1)
        return dp[i][j];
    int result = 0;
    if (s[i - 1] == t[j - 1])
    {
        int leave = numDistinctMemo(s, t, i - 1, j - 1, dp);
        int stay = numDistinctMemo(s, t, i - 1, j, dp);
        result = (leave + stay) % mod;
    }
    else
        result = numDistinctMemo(s, t, i - 1, j, dp);
    dp[i][j] = result;
    return result;
}
int numDistinctM(string s, string t)
{
    // Recursion
    int n = SZ(s);
    int m = SZ(t);
    VVI dp(n + 1, VI(m + 1, -1));
    return numDistinctMemo(s, t, n, m, dp);
}
// Optimal -----Tabulation----->
// Time Complexity: O(N*M)
// Reason: There are two nested loops
// Space Complexity: O(N*M)
// Reason: We are using an external array of size ‘N*M’. Stack Space is eliminated.
int numDistinctT(string &s1, string &s2)
{
    int n = SZ(s1);
    int m = SZ(s2);
    // Create a 2D DP array to store the count of distinct subsequences
    vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));

    // Initialize the first row: empty string s2 can be matched with any non-empty s1 in one way
    for (int i = 0; i <= n; i++)
    {
        dp[i][0] = 1;
    }

    // Initialize the first column: s1 can't match any non-empty s2
    for (int i = 1; i <= m; i++)
    {
        dp[0][i] = 0;
    }

    // Fill in the DP array
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            if (s1[i - 1] == s2[j - 1])
            {
                // If the characters match, we have two options:
                // 1. Match the current characters and move diagonally (dp[i-1][j-1])
                // 2. Leave the current character in s1 and match s2 with the previous characters (dp[i-1][j])
                dp[i][j] = (dp[i - 1][j - 1] + dp[i - 1][j]) % mod;
            }
            else
            {
                // If the characters don't match, we can only leave the current character in s1
                dp[i][j] = dp[i - 1][j];
            }
        }
    }

    // The value at dp[n][m] contains the count of distinct subsequences
    return dp[n][m];
}

// Most Optimal -----Space Optimization----->
// Time Complexity: O(N*M)
// Reason: There are two nested loops.
// Space Complexity: O(M)
// Reason: We are using an external array of size ‘M+1’ to store only one row.
int numDistinctSO(string &s1, string &s2)
{
    int n = SZ(s1);
    int m = SZ(s2);
    // Create an array to store the count of distinct subsequences for each character in s2
    vector<int> prev(m + 1, 0);

    // Initialize the count for an empty string (base case)
    prev[0] = 1;

    // Iterate through s1 and s2 to calculate the counts
    for (int i = 1; i <= n; i++)
    {
        for (int j = m; j >= 1; j--)
        { // Iterate in reverse direction to avoid overwriting values prematurely
            if (s1[i - 1] == s2[j - 1])
            {
                // If the characters match, we have two options:
                // 1. Match the current characters and add to the previous count (prev[j-1])
                // 2. Leave the current character in s1 and match s2 with the previous characters (prev[j])
                prev[j] = (prev[j - 1] + prev[j]) % mod;
            }
            // No need for an else statement since we can simply leave the previous count as is
        }
    }

    // The value at prev[m] contains the count of distinct subsequences
    return prev[m];
}

/*
33. Edit Distance
ANS : Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.
You have the following three operations permitted on a word:
Insert a character
Delete a character
Replace a character
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC : Exponentials in nature
// SC : O(n+m)
/*
Intuition : If character is matching then i--,j-- else not matched then
three choices we have:

return 1+f(i-1,j) // Insertion of character.
return 1+f(i,j-1) // Deletion of character.
return 1+f(i-1,j-1) // Replacement of character.

After that rerurn min of all cz we only need minimum operations.
*/
int editDistacnceRecr(int i, int j, string &word1, string &word2)
{
    // Base case :
    if (i < 0)
        return j + 1;
    if (j < 0)
        return i + 1;
    int ans = 0;
    if (word1[i] == word2[j])
    {
        ans = 0 + editDistacnceRecr(i - 1, j - 1, word1, word2);
    }
    else
    {
        // Minimum of three choices:
        // 1. Replace the character at S1[i] with the character at S2[j]
        // 2. Delete the character at S1[i]
        // 3. Insert the character at S2[j] into S1
        ans = 1 + min(editDistacnceRecr(i - 1, j - 1, word1, word2),
                      min(editDistacnceRecr(i - 1, j, word1, word2),
                          editDistacnceRecr(i, j - 1, word1, word2)));
    }
    return ans;
}
int editDistanceR(string word1, string word2)
{
    int n = SZ(word1);
    int m = SZ(word2);
    return editDistacnceRecr(n - 1, m - 1, word1, word2);
}
// Better -----Memoization------>
// Time Complexity: O(N*M)
// Reason: There are N*M states therefore at max ‘N*M’ new problems will be solved.
// Space Complexity: O(N*M) + O(N+M)
// Reason: We are using a recursion stack space(O(N+M)) and a 2D array ( O(N*M)).
// # HERE WE DO 1-BASED INDEXING SO THAT NO NEGATIVE IN VAULT
int editDistacnceMemo(int i, int j, string &word1, string &word2, VVI &dp)
{
    // Base case :
    if (i == 0)
        return j;
    if (j == 0)
        return i;
    if (dp[i][j] != -1)
        return dp[i][j];
    int ans = 0;
    if (word1[i - 1] == word2[j - 1])
    {
        ans = 0 + editDistacnceMemo(i - 1, j - 1, word1, word2, dp);
    }
    else
    {
        // Minimum of three choices:
        // 1. Replace the character at S1[i] with the character at S2[j]
        // 2. Delete the character at S1[i]
        // 3. Insert the character at S2[j] into S1
        ans = 1 + min(editDistacnceMemo(i - 1, j - 1, word1, word2, dp),
                      min(editDistacnceMemo(i - 1, j, word1, word2, dp),
                          editDistacnceMemo(i, j - 1, word1, word2, dp)));
    }
    return dp[i][j] = ans;
}
int editDistanceM(string word1, string word2)
{
    int n = SZ(word1);
    int m = SZ(word2);
    VVI dp(n + 1, VI(m + 1, -1));
    return editDistacnceMemo(n, m, word1, word2, dp);
}
// Optimal -----Tabulation----->
// Time Complexity: O(N*M)
// Reason: There are two nested loops
// Space Complexity: O(N*M)
// Reason: We are using an external array of size ‘N*M’. Stack Space is eliminated.
int editDistanceT(string &S1, string &S2)
{
    int n = S1.size();
    int m = S2.size();

    // Create a DP table to store edit distances
    vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));

    // Initialize the first row and column
    for (int i = 0; i <= n; i++)
    {
        dp[i][0] = i;
    }
    for (int j = 0; j <= m; j++)
    {
        dp[0][j] = j;
    }

    // Fill in the DP table
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            if (S1[i - 1] == S2[j - 1])
            {
                // If the characters match, no additional cost
                dp[i][j] = dp[i - 1][j - 1];
            }
            else
            {
                // Minimum of three choices:
                // 1. Replace the character at S1[i-1] with S2[j-1]
                // 2. Delete the character at S1[i-1]
                // 3. Insert the character at S2[j-1] into S1
                dp[i][j] = 1 + min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1]));
            }
        }
    }

    // The value at dp[n][m] contains the edit distance
    return dp[n][m];
}
// Most Optimal -----Space Optimization----->
// Time Complexity: O(N*M)
// Reason: There are two nested loops.
// Space Complexity: O(M)
// Reason: We are using an external array of size ‘M+1’ to store two rows.
int editDistanceSO(string &S1, string &S2)
{
    int n = S1.size();
    int m = S2.size();

    // Create two arrays to store previous and current row of edit distances
    vector<int> prev(m + 1, 0);
    vector<int> cur(m + 1, 0);

    // Initialize the first row
    for (int j = 0; j <= m; j++)
    {
        prev[j] = j;
    }

    // Calculate edit distances row by row
    for (int i = 1; i <= n; i++)
    {
        cur[0] = i; // Initialize the first column of the current row
        for (int j = 1; j <= m; j++)
        {
            if (S1[i - 1] == S2[j - 1])
            {
                // If the characters match, no additional cost
                cur[j] = prev[j - 1];
            }
            else
            {
                // Minimum of three choices:
                // 1. Replace the character at S1[i-1] with S2[j-1]
                // 2. Delete the character at S1[i-1]
                // 3. Insert the character at S2[j-1] into S1
                cur[j] = 1 + min(prev[j - 1], min(prev[j], cur[j - 1]));
            }
        }
        prev = cur; // Update the previous row with the current row
    }

    // The value at cur[m] contains the edit distance
    return cur[m];
}

/*
34. Wildcard Matching
ANS : Given a text and a wildcard pattern of size N and M respectively, implement a wildcard pattern matching algorithm that finds if the wildcard pattern is matched with the text. The matching should cover the entire text not partial text.
The wildcard pattern can include the characters ‘?’ and ‘*’
 ‘?’ – matches any single character
 ‘*’ – Matches any sequence of characters(sequence can be of length 0 or more)
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC : Exponential in nature
// SC :O(N+M)
bool wildcardMatchingRecr(int i, int j, string &p, string &t)
{
    if (i < 0 && j < 0)
        return true;
    if (i < 0 && j >= 0)
        return false;
    ;
    if (j < 0 && i >= 0)
    {
        FORE(ii, i)
        {
            if (p[ii] != '*')
                return false;
        }
        return true;
    }
    if (p[i] == t[j] || p[i] == '?')
        return wildcardMatchingRecr(i - 1, j - 1, p, t);
    if (p[i] == '*')
    {
        return wildcardMatchingRecr(i - 1, j, p, t) || wildcardMatchingRecr(i, j - 1, p, t);
    }
    return false;
}
bool wildcardMatchingR(string p, string t)
{
    int n = SZ(p);
    int m = SZ(t);
    return wildcardMatchingRecr(n - 1, m - 1, p, t);
}

// Better -----Memoization------>
// Time Complexity: O(N*M)
// Reason: There are N*M states therefore at max ‘N*M’ new problems will be solved.
// Space Complexity: O(N*M) + O(N+M)
// Reason: We are using a recursion stack space(O(N+M)) and a 2D array ( O(N*M)).

bool wildcardMatchingMemo(int i, int j, string &p, string &t, VVI &dp)
{
    if (i == 0 && j == 0)
        return true;
    if (i == 0 && j > 0)
        return false;
    ;
    if (j == 0 && i > 0)
    {
        FORE(ii, i)
        {
            if (p[ii - 1] != '*')
                return false;
        }
        return true;
    }
    if (dp[i][j] != -1)
        return dp[i][j];
    if (p[i - 1] == t[j - 1] || p[i - 1] == '?')
        return dp[i][j] = wildcardMatchingMemo(i - 1, j - 1, p, t, dp);
    if (p[i - 1] == '*')
    {
        return dp[i][j] = wildcardMatchingMemo(i - 1, j, p, t, dp) || wildcardMatchingMemo(i, j - 1, p, t, dp);
    }
    return false;
}
bool wildcardMatchingM(string p, string t)
{
    int n = SZ(p);
    int m = SZ(t);
    VVI dp(n + 1, VI(m + 1, -1));
    return wildcardMatchingMemo(n, m, p, t, dp); // 1-based indexing
}
// Optimal -----Tabulation----->
// Time Complexity: O(N*M)
// Reason: There are two nested loops
// Space Complexity: O(N*M)
// Reason: We are using an external array of size ‘N*M’. Stack Space is eliminated.
bool isAllStars(string &S1, int i)
{
    // S1 is taken in 1-based indexing
    for (int j = 1; j <= i; j++)
    {
        if (S1[j - 1] != '*')
            return false;
    }
    return true;
}
bool wildcardMatchingT(string S1, string S2)
{
    int n = S1.size();
    int m = S2.size();

    // Create a DP table to memoize results
    vector<vector<bool>> dp(n + 1, vector<bool>(m, false));

    // Initialize the first row and column
    dp[0][0] = true;
    for (int j = 1; j <= m; j++)
    {
        dp[0][j] = false;
    }
    for (int i = 1; i <= n; i++)
    {
        dp[i][0] = isAllStars(S1, i);
    }

    // Fill in the DP table
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            if (S1[i - 1] == S2[j - 1] || S1[i - 1] == '?')
            {
                dp[i][j] = dp[i - 1][j - 1];
            }
            else
            {
                if (S1[i - 1] == '*')
                {
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
                }
                else
                {
                    dp[i][j] = false;
                }
            }
        }
    }

    // The value at dp[n][m] contains whether S1 matches S2
    return dp[n][m];
}
// Most Optimal -----Space Optimization----->
// Time Complexity: O(N*M)
// Reason: There are two nested loops.
// Space Complexity: O(M)
// Reason: We are using an external array of size ‘M+1’ to store two rows.
bool wildcardMatchingSO(string &S1, string &S2)
{
    int n = S1.size();
    int m = S2.size();

    // Create two arrays to store previous and current rows of matching results
    vector<bool> prev(m + 1, false);
    vector<bool> cur(m + 1, false);

    prev[0] = true; // Initialize the first element of the previous row to true

    for (int i = 1; i <= n; i++)
    {
        cur[0] = isAllStars(S1, i); // Initialize the first element of the current row
        for (int j = 1; j <= m; j++)
        {
            if (S1[i - 1] == S2[j - 1] || S1[i - 1] == '?')
            {
                cur[j] = prev[j - 1]; // Characters match or S1 has '?'
            }
            else
            {
                if (S1[i - 1] == '*')
                {
                    cur[j] = prev[j] || cur[j - 1]; // '*' represents empty or a character
                }
                else
                {
                    cur[j] = false; // Characters don't match and S1[i-1] is not '*'
                }
            }
        }
        prev = cur; // Update the previous row with the current row
    }

    // The value at prev[m] contains whether S1 matches S2
    return prev[m];
}

/*##############################DP ON STOCKS#################################*/
/*
35. Best Time to Buy and Sell Stock
ANS :  You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.
Input :   || Output :
*/
// Bruteforce ------Recursion----->
// TC :O(N)
// SC :O(1)
/*
Intuition : If you're selling it on i-th day,you buy on the minimum price from 1st->(i-1)
*/
int maxProfitI(vector<int> &prices)
{
    int n = SZ(prices);
    if (n == 1)
        return 0;
    int mini = prices[0];
    int maxProfit = 0;
    for (int i = 1; i < n; i++)
    {
        int cost = prices[i] - mini;
        maxProfit = max(maxProfit, cost);
        mini = min(mini, prices[i]);
    };
    return maxProfit;
}
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
36. Buy and Sell Stock - II
ANS : You are given an integer array prices where prices[i] is the price of a given stock on the ith day.
On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. However, you can buy it then immediately sell it on the same day.
Find and return the maximum profit you can achieve.
Input :   || Output :
*/
// Bruteforce -----Recursion------>
// Time Complexity: O(2^N)
// Reason: We are running a for loop for ‘N’ times to calculate the total sum
// Space Complexity: O(N)
// Reason: We are using a recursion stack space(O(N))
/*
Intuition : Every day, we will have two choices, either to do nothing and move to the next day or to buy/sell (based on the last transaction) and find out the profit. Therefore we need to generate all the choices in order to compare the profit. As we need to try out all the possible choices, we will use recursion.
we need to respect the condition that we can’t buy a stock again, that is we need to first sell a stock, and then we can buy that again. Therefore we need a second variable ‘buy’ which tells us on a particular day whether we can buy or sell the stock.
Generate all choices :
Case 1: When buy == 0, we can buy the stock.

If we can buy the stock on a particular day, we have two options:

Option 1: To do no transaction and move to the next day. In this case, the net profit earned will be 0 from the current transaction, and to calculate the maximum profit starting from the next day, we will recursively call f(ind+1,0). As we have not bought the stock, the ‘buy’ variable value will still remain 0, indicating that we can buy the stock the next day.
Option 2: The other option is to buy the stock on the current day. In this case, the net profit earned from the current transaction will be -Arr[i]. As we are buying the stock, we are giving money out of our pocket, therefore the profit we earn is negative. To calculate the maximum profit starting from the next day, we will recursively call f(ind+1,1). As we have bought the stock, the ‘buy’ variable value will change to 1, indicating that we can’t buy and only sell the stock the next day.
Case 2: When buy == 1, we can sell the stock.

If we can buy the stock on a particular day, we have two options:

Option 1: To do no transaction and move to the next day. In this case, the net profit earned will be 0 from the current transaction, and to calculate the maximum profit starting from the next day, we will recursively call f(ind+1,1). As we have not bought the stock, the ‘buy’ variable value will still remain at 1, indicating that we can’t buy and only sell the stock the next day.
Option 2: The other option is to sell the stock on the current day. In this case, the net profit earned from the current transaction will be +Arr[i]. As we are selling the stock, we are putting the money into our pocket, therefore the profit we earn is positive. To calculate the maximum profit starting from the next day, we will recursively call f(ind+1,0). As we have sold the stock, the ‘buy’ variable value will change to 0, indicating that we can buy the stock the next day.

*/
int maxProfitRecr(int ind, VI &prices, int buy)
{
    int n = SZ(prices);
    // Base Case :
    if (ind == n)
        return 0;
    int profit = 0;
    if (buy)
    {
        // Max of pick and notPick
        // If we but on first day then we can sell with 0 if we not buy on first day then we can buy=1
        profit = max(-prices[ind] + maxProfitRecr(ind + 1, prices, 0), 0 + maxProfitRecr(ind + 1, prices, 1));
    }
    else
        profit = max(prices[ind] + maxProfitRecr(ind + 1, prices, 1), 0 + maxProfitRecr(ind + 1, prices, 0));
    return profit;
}
int maxProfitIIR(vector<int> &prices)
{
    int buy = 1;
    return maxProfitRecr(0, prices, buy);
}
// Better ------Memoization----->
// Time Complexity: O(N*2)
// Reason: There are N*2 states therefore at max ‘N*2’ new problems will be solved and we are running a for loop for ‘N’ times to calculate the total sum
// Space Complexity: O(N*2) + O(N)
// Reason: We are using a recursion stack space(O(N)) and a 2D array ( O(N*2)).
int maxProfitMemo(int ind, VI &prices, int buy, VVI &dp)
{
    int n = SZ(prices);
    // Base Case :
    if (ind == n)
        return 0;
    if (dp[ind][buy] != -1)
        return dp[ind][buy];
    int profit = 0;
    if (buy)
    {
        // Max of pick and notPick
        // If we but on first day then we can sell with 0 if we not buy on first day then we can buy=1
        profit = max(-prices[ind] + maxProfitMemo(ind + 1, prices, 0, dp), 0 + maxProfitMemo(ind + 1, prices, 1, dp));
    }
    else
        profit = max(prices[ind] + maxProfitMemo(ind + 1, prices, 1, dp), 0 + maxProfitMemo(ind + 1, prices, 0, dp));
    return dp[ind][buy] = profit;
}
int maxProfitIIM(vector<int> &prices)
{
    int n = SZ(prices);
    int buy = 1;
    VVI dp(n, VI(2, -1));
    return maxProfitMemo(0, prices, buy, dp);
}
// Optimal -----Tabulation----->
// Time Complexity: O(N*2)
// Reason: There are two nested loops that account for O(N*2) complexity.
// Space Complexity: O(N*2)
// Reason: We are using an external array of size ‘N*2’. Stack Space is eliminated.
/*
First go for base case then
check for changing params like ind and buy
Then copy the recursion
*/
int maxProfitIIT(vector<int> &prices)
{
    int n = SZ(prices);
    int buy = 1;
    VVI dp(n + 1, VI(2, 0));
    // Base case :
    dp[n][0] = dp[n][1] = 0; // As (ind==n) return 0
    // Changing params loop and here Tabulation is Buttom-Up
    for (int ind = n - 1; ind >= 0; ind--)
    {                                      // First changing param
        for (int buy = 0; buy <= 1; buy++) // Second changing param
        {
            int profit = 0;
            if (buy)
            {
                // Max of pick and notPick
                // If we but on first day then we can sell with 0 if we not buy on first day then we can buy=1
                profit = max(-prices[ind] + dp[ind + 1][0], 0 + dp[ind + 1][1]);
            }
            else
                profit = max(prices[ind] + dp[ind + 1][1], 0 + dp[ind + 1][0]);
            dp[ind][buy] = profit;
        }
    }
    return dp[0][1];
}
// Most Optimal -----Space Optimization----->
// TC :O(N*2)
// SC :O(1)
int maxProfitIISO(vector<int> &prices)
{
    int n = SZ(prices);
    VI ahead(2, 0), cur(2, 0);
    // Base case :
    ahead[0] = ahead[1] = 0; // As (ind==n) return 0
    // Changing params loop and here Tabulation is Buttom-Up
    for (int ind = n - 1; ind >= 0; ind--)
    {                                      // First changing param
        for (int buy = 0; buy <= 1; buy++) // Second changing param
        {
            int profit = 0;
            if (buy)
            {
                // Max of pick and notPick
                // If we but on first day then we can sell with 0 if we not buy on first day then we can buy=1
                profit = max(-prices[ind] + ahead[0], 0 + ahead[1]);
            }
            else
                profit = max(prices[ind] + ahead[1], 0 + ahead[0]);
            cur[buy] = profit;
        }
        ahead = cur;
    }
    return ahead[1];
}
// Most Optimal -----Variable Optimization----->
// TC :O(N*2)
// SC :O(1)
int maxProfitIIVO(vector<int> &prices)
{
    int n = SZ(prices);
    int aheadNotBuy, aheadBuy, curBuy, curNotBuy;
    aheadBuy = aheadNotBuy = 0;
    for (int ind = n - 1; ind >= 0; ind--)
    {
        curNotBuy = max(prices[ind] + aheadBuy, 0 + aheadNotBuy);
        curBuy = max(-prices[ind] + aheadNotBuy, 0 + aheadBuy);
        aheadBuy = curBuy;
        aheadNotBuy = curNotBuy;
    }
    return aheadBuy;
}

/*
37.  Best Time to Buy and Sell Stock III
ANS : You are given an array prices where prices[i] is the price of a given stock on the ith day.

Find the maximum profit you can achieve. You may complete at most two transactions.

Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
Input :   || Output :
*/
// Bruteforce -----Recursion------>
// TC : O(2^N)
// SC :O(N) Recursion stack space
int maxProfitIIIRecr(int ind, VI &prices, int buy, int cap)
{
    int n = SZ(prices);
    // Base Case :
    if (ind == n)
        return 0;
    if (cap == 0)
        return 0;
    int profit = 0;
    if (buy)
    {
        // Max of pick and notPick
        // If we but on first day then we can sell with 0 if we not buy on
        // first day then we can buy=1
        profit = max(-prices[ind] + maxProfitIIIRecr(ind + 1, prices, 0, cap),
                     0 + maxProfitIIIRecr(ind + 1, prices, 1, cap));
    }
    else
        profit =
            max(prices[ind] + maxProfitIIIRecr(ind + 1, prices, 1, cap - 1),
                0 + maxProfitIIIRecr(ind + 1, prices, 0, cap));
    return profit;
}
int maxProfitIIIR(vector<int> &prices)
{
    int buy = 1, cap = 2;
    return maxProfitIIIRecr(0, prices, buy, cap);
}
// Better ------Memoization----->
// Time Complexity: O(N*2*3)
// Reason: There are N*2*3 states therefore at max ‘N*2*3’ new problems will be solved.
// Space Complexity: O(N*2*3) + O(N)
// Reason: We are using a recursion stack space(O(N)) and a 3D array ( O(N*2*3)).
int maxProfitIIIMemo(int ind, VI &prices, int buy, int cap, VVVI &dp)
{
    int n = SZ(prices);
    // Base Case :
    if (ind == n)
        return 0;
    if (cap == 0)
        return 0;
    if (dp[ind][buy][cap] != -1)
        return dp[ind][buy][cap];
    int profit = 0;
    if (buy)
    {
        // Max of pick and notPick
        // If we but on first day then we can sell with 0 if we not buy on
        // first day then we can buy=1
        profit = max(-prices[ind] + maxProfitIIIMemo(ind + 1, prices, 0, cap, dp),
                     0 + maxProfitIIIMemo(ind + 1, prices, 1, cap, dp));
    }
    else
        profit =
            max(prices[ind] + maxProfitIIIMemo(ind + 1, prices, 1, cap - 1, dp),
                0 + maxProfitIIIMemo(ind + 1, prices, 0, cap, dp));
    return dp[ind][buy][cap] = profit;
}
int maxProfitIIIM(vector<int> &prices)
{
    int n = SZ(prices);
    int buy = 1, cap = 2;
    vector<vector<vector<int>>> dp(n, VVI(2, VI(3, -1)));
    return maxProfitIIIMemo(0, prices, buy, cap, dp);
}
// Optimal -----Tabulation----->
// Time Complexity: O(N*2*3)
// Reason: There are three nested loops that account for O(N*2*3) complexity.
// Space Complexity: O(N*2*3)
// Reason: We are using an external array of size ‘N*2*3’. Stack Space is eliminated.
int maxProfitIIIT(vector<int> &prices)
{
    int n = SZ(prices);
    vector<vector<vector<int>>> dp(n + 1, VVI(2, VI(3, 0)));
    for (int ind = n - 1; ind >= 0; ind--)
    { // First changing param
        for (int buy = 0; buy <= 1; buy++)
        { // Second changing param
            for (int cap = 1; cap <= 2; cap++)
            { // Third changing param
                if (buy)
                {
                    // Max of pick and notPick
                    // If we but on first day then we can sell with 0 if we not buy on
                    // first day then we can buy=1
                    dp[ind][buy][cap] = max(-prices[ind] +
                                                dp[ind + 1][0][cap],
                                            0 + dp[ind + 1][1][cap]);
                }
                else
                    dp[ind][buy][cap] = max(prices[ind] + dp[ind + 1][1][cap - 1],
                                            0 + dp[ind + 1][0][cap]);
            }
        }
    }
    return dp[0][1][2];
}

// Most Optimal -----Space Optimization----->
// Time Complexity: O(N*2*3)
// Reason: There are three nested loops that account for O(N*2*3) complexity
// Space Complexity: O(1)
int maxProfitIIISO(vector<int> &prices)
{
    int n = SZ(prices);
    VVI after(2, VI(3, 0));
    VVI cur(2, VI(3, 0));
    for (int ind = n - 1; ind >= 0; ind--)
    { // First changing param
        for (int buy = 0; buy <= 1; buy++)
        { // Second changing param
            for (int cap = 1; cap <= 2; cap++)
            { // Third changing param
                if (buy)
                {
                    // Max of pick and notPick
                    // If we but on first day then we can sell with 0 if we not buy on
                    // first day then we can buy=1
                    cur[buy][cap] = max(-prices[ind] +
                                            after[0][cap],
                                        0 + after[1][cap]);
                }
                else
                    cur[buy][cap] = max(prices[ind] + after[1][cap - 1],
                                        0 + after[0][cap]);
            }
        }
        after = cur;
    }
    return after[1][2];
}

/*
38. Best Time to Buy and Sell Stock IV
ANS : You are given an integer array prices where prices[i] is the price of a given stock on the ith day, and an integer k.
Find the maximum profit you can achieve. You may complete at most k transactions: i.e. you may buy at most k times and sell at most k times.
Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
Input :   || Output :
*/
/*
Intuition : Instedof doing buy/sell and capacity calculation we can do using transaction
*/
// Bruteforce -----Recursion------>
// TC : O(2^(n * 2k))
// SC :O(N) recursion stack space
int maxProfitIIItRecr(int ind, int trans, VI &prices, int k)
{
    int n = SZ(prices);
    // Base case :
    if (ind == n || trans == 2 * k)
        return 0;
    if (trans % 2 == 0) // Buy
    {
        return max(-prices[ind] + maxProfitIIItRecr(ind + 1, trans + 1, prices, k),
                   0 + maxProfitIIItRecr(ind + 1, trans, prices, k));
    }
    else
        return max(prices[ind] + maxProfitIIItRecr(ind + 1, trans + 1, prices, k),
                   0 + maxProfitIIItRecr(ind + 1, trans, prices, k));
}
int maxProfitIIItR(vector<int> &prices, int k)
{
    return maxProfitIIItRecr(0, 0, prices, k); // Index,Transaction count, Array, Limit
}
// Better ------Memoization----->
// Time Complexity: O(n * k)
// Space Complexity: O(n * k)
int maxProfitIIItMemo(int ind, VI &prices, int t, VVI &dp, int k)
{
    int n = SZ(prices);
    // Base case :
    if (ind == n || t == 2 * k)
        return 0;
    if (dp[ind][t] != -1)
        return dp[ind][t];
    if (t % 2 == 0) // Buy
    {
        return dp[ind][t] = max(-prices[ind] + maxProfitIIItMemo(ind + 1, prices, t + 1, dp, k),
                                0 + maxProfitIIItMemo(ind + 1, prices, t, dp, k));
    }
    else
        return dp[ind][t] = max(prices[ind] + maxProfitIIItMemo(ind + 1, prices, t + 1, dp, k),
                                0 + maxProfitIIItMemo(ind + 1, prices, t, dp, k));
}
int maxProfitIIItM(vector<int> &prices, int k)
{
    int n = SZ(prices);
    VVI dp(n, VI(2 * k, -1));
    return maxProfitIIItMemo(0, prices, 0, dp, k);
}
// Optimal -----Tabulation----->
// Time Complexity: O(n * k)
// Space Complexity: O(n * k)
int maxProfitIIItT(vector<int> &prices, int k)
{
    int n = SZ(prices);
    VVI dp(n + 1, VI(2 * k + 1, 0)); // We're adding extra space to avoid going out of bounds, as we're accessing indices ind+1 and t+1 in our calculations.
    for (int ind = n - 1; ind >= 0; ind--)
    { // First changing param
        for (int t = 2 * k - 1; t >= 0; t--)
        { // Second changing param

            if (t % 2 == 0)
            {
                // Max of pick and notPick
                // If we but on first day then we can sell with 0 if we not buy on
                // first day then we can buy=1
                dp[ind][t] = max(-prices[ind] +
                                     dp[ind + 1][t + 1],
                                 0 + dp[ind + 1][t]);
            }
            else
                dp[ind][t] = max(prices[ind] + dp[ind + 1][t + 1],
                                 0 + dp[ind + 1][t]);
        }
    }
    return dp[0][0];
}
// Most Optimal -----Space Optimization----->
// Time Complexity: O(n * k)
// Space Complexity: O(k)
int maxProfitIIItSO(vector<int> &prices, int k)
{
    int n = SZ(prices);
    // We're adding extra space to avoid going out of bounds, as we're accessing indices ind+1 and t+1 in our calculations.
    VI after(2 * k + 1, 0);
    VI cur(2 * k + 1, 0);
    for (int ind = n - 1; ind >= 0; ind--)
    { // First changing param
        for (int t = 2 * k - 1; t >= 0; t--)
        { // Second changing param

            if (t % 2 == 0)
            {
                // Max of pick and notPick
                // If we but on first day then we can sell with 0 if we not buy on
                // first day then we can buy=1
                cur[t] = max(-prices[ind] +
                                 after[t + 1],
                             0 + after[t]);
            }
            else
                cur[t] = max(prices[ind] + after[t + 1],
                             0 + after[t]);
        }
        after = cur;
    }
    return after[0];
}

/*
39. Best Time to Buy and Sell Stock with Cooldown
ANS : You are given an array prices where prices[i] is the price of a given stock on the ith day.
Find the maximum profit you can achieve. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times) with the following restrictions:
After you sell your stock, you cannot buy stock on the next day (i.e., cooldown one day).
Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
Input :   || Output :
*/
// Bruteforce -----Recursion------>
// Time Complexity: O(2^N)
// Reason: We are running a for loop for ‘N’ times to calculate the total sum
// Space Complexity: O(N)
// Reason: We are using a recursion stack space(O(N))
int maxProfitCoolRecr(int ind, VI &prices, int buy)
{
    int n = SZ(prices);
    // Base Case :
    if (ind >= n) // Here we do >= as we're  doing ind+2 assume that we're at ind==n-1 so thats why
        return 0;
    int profit = 0;
    if (buy)
    {
        // Max of pick and notPick
        // If we but on first day then we can sell with 0 if we not buy on
        // first day then we can buy=1
        profit = max(-prices[ind] + maxProfitCoolRecr(ind + 1, prices, 0),
                     0 + maxProfitCoolRecr(ind + 1, prices, 1));
    }
    else
        profit = max(prices[ind] + maxProfitCoolRecr(ind + 2, prices, 1),
                     0 + maxProfitCoolRecr(ind + 1, prices, 0));
    return profit;
}

int maxProfitCoolR(vector<int> &prices)
{
    int buy = 1;
    return maxProfitCoolRecr(0, prices, buy);
}
// Better ------Memoization----->
// Time Complexity: O(N*2)
// Reason: There are N*2 states therefore at max ‘N*2’ new problems will be solved and we are running a for loop for ‘N’ times to calculate the total sum
// Space Complexity: O(N*2) + O(N)
// Reason: We are using a recursion stack space(O(N)) and a 2D array ( O(N*2)).
int maxProfitCoolMemo(int ind, VI &prices, int buy, VVI &dp)
{
    int n = SZ(prices);
    // Base Case :
    if (ind >= n)
        return 0;
    if (dp[ind][buy] != -1)
        return dp[ind][buy];
    int profit = 0;
    if (buy)
    {
        // Max of pick and notPick
        // If we but on first day then we can sell with 0 if we not buy on first day then we can buy=1
        profit = max(-prices[ind] + maxProfitCoolMemo(ind + 1, prices, 0, dp), 0 + maxProfitCoolMemo(ind + 1, prices, 1, dp));
    }
    else
        profit = max(prices[ind] + maxProfitCoolMemo(ind + 2, prices, 1, dp), 0 + maxProfitCoolMemo(ind + 1, prices, 0, dp));
    return dp[ind][buy] = profit;
}
int maxProfitCoolM(vector<int> &prices)
{
    int n = SZ(prices);
    int buy = 1;
    VVI dp(n, VI(2, -1));
    return maxProfitCoolMemo(0, prices, buy, dp);
}
// Optimal -----Tabulation----->
// Time Complexity: O(N*2)
// Reason: There are two nested loops that account for O(N*2) complexity.
// Space Complexity: O(N*2)
// Reason: We are using an external array of size ‘N*2’. Stack Space is eliminated.
int maxProfitCoolT(vector<int> &prices)
{
    int n = SZ(prices);
    int buy = 1;
    VVI dp(n + 2, VI(2, 0));
    // Changing params loop and here Tabulation is Buttom-Up
    for (int ind = n - 1; ind >= 0; ind--)
    { // First changing param
      // Second changing param
      // Instedof doing loop you can direct access buy like :
        dp[ind][1] = max(-prices[ind] + dp[ind + 1][0], 0 + dp[ind + 1][1]);

        dp[ind][0] = max(prices[ind] + dp[ind + 2][1], 0 + dp[ind + 1][0]);
    }
    return dp[0][1];
}
// Most Optimal -----Space Optimization----->
// TC :O(N)
// SC :O(6) much like constant
// As w're doing ind+1,ind+2 we
int maxProfitCoolSO(vector<int> &prices)
{
    int n = SZ(prices);
    VI ahead(3, 0), cur(3, 0);
    VI f1(2, 0);
    VI f2(2, 0);
    VI fcur(2, 0);
    // Changing params loop and here Tabulation is Buttom-Up
    for (int ind = n - 1; ind >= 0; ind--)
    { // First changing param
        cur[1] = max(-prices[ind] + f1[0], 0 + f1[1]);
        cur[0] = max(prices[ind] + f2[1], 0 + f1[1]);
        f2 = f1;
        f1 = cur;
    }
    return cur[1];
}

/*
40. Best Time to Buy and Sell Stock with Transaction Fee
ANS : You are given an array prices where prices[i] is the price of a given stock on the ith day, and an integer fee representing a transaction fee.

Find the maximum profit you can achieve. You may complete as many transactions as you like, but you need to pay the transaction fee for each transaction.

Note:

You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
The transaction fee is only charged once for each stock purchase and sale.
Input :   || Output :
*/
// Bruteforce -----Recursion------>
// TC :
// SC :
// Better ------Memoization----->
// TC :
// SC :
// Optimal -----Tabulation----->
// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :O(N)
// SC :O(1)
// You can give fee when you're buing or selling up to you but remember for per transaction you have to give fee
int maxProfitFeeR(vector<int> &prices, int fee)
{
    int n = SZ(prices);
    int aheadNotBuy, aheadBuy, curBuy, curNotBuy;
    aheadBuy = aheadNotBuy = 0;
    for (int ind = n - 1; ind >= 0; ind--)
    {
        // sell
        curNotBuy = max(prices[ind] - fee + aheadBuy, 0 + aheadNotBuy);
        // buy
        curBuy = max(-prices[ind] + aheadNotBuy, 0 + aheadBuy);
        aheadBuy = curBuy;
        aheadNotBuy = curNotBuy;
    }
    return aheadBuy;
}
/*##############################DP ON LIS#################################*/
/*
41. Longest Increasing Subsequence
ANS : Given an integer array nums, return the length of the longest strictly increasing  subsequence.Qs says thats its increasing so if array elems are same then return 1
Input :   || Output :
*/
// Very bruteforce is generate all subsequences then check for increasing then check for longest.It'll givr e you O(2^n) in exponential in nature
// Bruteforce -----Recursion------>
// TC :O(2^n) cz in every index we've 2 options take or not take
// SC :O(N) for recursion stack space
// As we are taking increasing values so i have to remember the last one(index) i pick so that i can pick next bigger value from the prev
int lengthOfLISRecr(int ind, int prevInd, VI &nums)
{
    int n = SZ(nums);
    // Base case :
    if (ind == n)
        return 0;
    // notTake
    int notTake = 0 + lengthOfLISRecr(ind + 1, prevInd, nums);
    int take = 0;
    if (prevInd == -1 || nums[ind] > nums[prevInd])
    {
        take = 1 + lengthOfLISRecr(ind + 1, ind, nums);
    }
    return max(notTake, take);
}
int lengthOfLISR(vector<int> &nums)
{
    return lengthOfLISRecr(0, -1, nums); // len,prevInd,array
}
// Better ------Memoization----->
// TC :O(NxN)
// SC :O(NxN)+O(N)
int lengthOfLISMemo(int ind, int prevInd, VI &nums, VVI &dp)
{
    int n = SZ(nums);
    // Base case :
    if (ind == n)
        return 0;
    if (dp[ind][prevInd + 1] != -1)
        return dp[ind][prevInd + 1];
    // notTake
    int notTake = 0 + lengthOfLISMemo(ind + 1, prevInd, nums, dp);
    int take = 0;
    if (prevInd == -1 || nums[ind] > nums[prevInd])
    {
        take = 1 + lengthOfLISMemo(ind + 1, ind, nums, dp);
    }
    return dp[ind][prevInd + 1] = max(notTake, take);
}
int lengthOfLISM(vector<int> &nums)
{
    int n = SZ(nums);
    VVI dp(n, VI(n + 1, -1));
    return lengthOfLISMemo(0, -1, nums, dp); // len,prevInd,array
    // As we can't save -1 ind in dp so we use cordinate change means we're trying prevInd=-1 to n-1 so insteadof -1 we use 0 for 0-> 1,1->2 and so on.........
}
// Optimal -----Tabulation----->
// TC :O(n^2)
// SC :O(n^2)
int lengthOfLIST(vector<int> &nums)
{
    int n = SZ(nums);
    VVI dp(n + 1, VI(n + 1, 0));
    for (int ind = n - 1; ind >= 0; ind--)
    { // First changing param
        for (int prevInd = ind - 1; prevInd >= -1; prevInd--)
        { // Second changing param(As we know prev value is refer to ind-1)
          // notTake
            int notTake = 0 + dp[ind + 1][prevInd + 1];
            int take = 0;
            if (prevInd == -1 || nums[ind] > nums[prevInd])
            {
                take = 1 + dp[ind + 1][ind + 1];
            }
            dp[ind][prevInd + 1] = max(notTake, take);
        }
    }
    return dp[0][-1 + 1];
}
// Most Optimal -----Space Optimization----->
// TC :O(N^2)
// SC :O(nx2)
int lengthOfLISSO(vector<int> &nums)
{
    int n = SZ(nums);
    VI next(n + 1, 0), cur(n + 1, 0);
    for (int ind = n - 1; ind >= 0; ind--)
    {
        for (int prevInd = ind - 1; prevInd >= -1; prevInd--)
        {
            int notPick = 0 + next[prevInd + 1];
            int pick = 0;
            if (prevInd == -1 || nums[ind] > nums[prevInd])
            {
                pick = 1 + next[ind + 1];
            }
            cur[prevInd + 1] = max(notPick, pick);
        }
        next = cur;
    }
    return next[-1 + 1];
}
// Most Most Optimal -----Optimal Tabulation----->
// TC :O(N^2)
// SC :O(N)
/*
For every index i of the array ‘arr’;
dp[ i ] is the length of the longest increasing subsequence that is possible that end with index ind of the original array.
*/
int lengthOfLISOT(vector<int> &nums)
{
    int n = SZ(nums);
    VI dp(n, 1);
    int maxi = 1;
    for (int i = 0; i < n; i++)
    {
        for (int prev = 0; prev < i; prev++)
        {
            if (nums[prev] < nums[i])
            {
                dp[i] = max(dp[i], 1 + dp[prev]);
            }
        }
        maxi = max(maxi, dp[i]);
    }
    return maxi;
}
/*
42. Print Longest Increasing Subsequence
ANS : Given an integer n and an array of integers arr, return the Longest Increasing Subsequence which is Index-wise lexicographically smallest.
Input :   || Output :
*/
// Bruteforce -----Trace Back Algorithmic approach------>
// TC : O(N*N)
// SC :O(N)  We are only using two rows of size ‘N’. and O(N) for storing the ans
VI longestIncreasingSubsequenceTB(int n, vector<int> &nums)
{
    VI dp(n, 1), hash(n);
    int maxi = 1, lastInd = 0;
    for (int i = 0; i < n; i++)
    {
        hash[i] = i; // initializing with current index
        for (int prev = 0; prev < i; prev++)
        {
            if (nums[prev] < nums[i] &&
                1 + dp[prev] > dp[i])
            {
                dp[i] = 1 + dp[prev];
                hash[i] = prev;
            }
        }
        if (dp[i] > maxi)
        {
            maxi = dp[i];
            lastInd = i;
        }
    }
    VI ans;
    ans.PB(nums[lastInd]);
    while (hash[lastInd] != lastInd)
    {
        lastInd = hash[lastInd];
        ans.PB(nums[lastInd]);
    }
    REV(ans);
    return ans;
}
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
43. Print Longest Increasing Subsequence || Binary Search
↑↑ Upto here we solve TC : O(N^2) SC O(N) but constrains is O(10^5) as our TC is N^2 so its became O(10^10) Now optimize it
ANS :
Input :   || Output :
*/
/*
Intuition : is so simple go through all arr[i] if its same with GS(generated subsequences) then replace with same elem else if its not matched then place its right position means place it in a increasing order
& as we know lower_bound() gives us
The element X itself, if it is present.
Or the next largest element, if the element is not present.
GS=Creating a temp array
*/

// Most Optimal -----Time Optimization----->
// Time Complexity: O(N*logN)
// Reason: We iterate over the array of size N and in every iteration, we perform a binary search which takes logN time.
// Space Complexity: O(N)
// Reason: We are using an extra array of size N to store the temp variable.
int longestSubsequence(int n, VI &arr)
{
    //   Using Binary search
    // Init a temp array to generate subsequence
    VI temp;
    temp.PB(arr[0]); // Put the 1st elem
    int len = 1;
    for (int i = 1; i < n; i++)
    {
        if (arr[i] > temp.back())
        { // If we found greater then push to next of temp[i]
            temp.PB(arr[i]);
            len++;
        }
        else
        {
            // finding insert position in a increasing order
            int ind = lower_bound(temp.begin(), temp.end(), arr[i]) - temp.begin(); // Retuens index
            temp[ind] = arr[i];
        }
    }
    return len;
}

/*
44. Largest Divisible Subset
ANS : Given a set of distinct positive integers nums, return the largest subset answer such that every pair (answer[i], answer[j]) of elements in this subset satisfies:
answer[i] % answer[j] == 0, or
answer[j] % answer[i] == 0
If there are multiple solutions, return any of them.
Input :   || Output :
*/
// Bruteforce -----Recursion------>
// TC :
// SC :
/*
Intuition : First sort the elems then pick first elem and add to hash array to save previous index now check from arr[1]->arr[n] if arr[i]%arr[prevInd]==0
Why prevInd cz if array contain [1,4,8,16] that and i save prevInd as 2=>8 and i check 8%4==0 as on 4%1==0 as on 8%1==0
As we know "If a divisor divides a dividend, then the dividend can be divided by the divisor."
*/
// Better ------Memoization----->
// Time Complexity: O(N*N)
// Reason: There are two nested loops.
// Space Complexity: O(N)
// Reason: We are only using two rows of size n.
vector<int> largestDivisibleSubset(vector<int> &nums)
{
    int n = SZ(nums);
    SORT(nums);
    VI dp(n, 1), hash(n);
    int maxi = 1, lastInd = 0;
    for (int i = 0; i < n; i++)
    {
        hash[i] = i;
        for (int prev = 0; prev < i; prev++)
        {
            if (nums[i] % nums[prev] == 0 && 1 + dp[prev] > dp[i])
            {
                dp[i] = 1 + dp[prev];
                hash[i] = prev;
            }
        }
        if (dp[i] > maxi)
        {
            maxi = dp[i];
            lastInd = i;
        }
    }
    VI ans;
    ans.PB(nums[lastInd]);
    while (hash[lastInd] != lastInd)
    {
        lastInd = hash[lastInd];
        ans.PB(nums[lastInd]);
    }
    REV(ans);
    return ans;
}
// Optimal -----Tabulation----->
// TC :
// SC :
// Most Optimal -----Space Optimization----->
// TC :
// SC :

/*
45. Longest String Chain
ANS : You are given an array of words where each word consists of lowercase English letters.

wordA is a predecessor of wordB if and only if we can insert exactly one letter anywhere in wordA without changing the order of the other characters to make it equal to wordB.

For example, "abc" is a predecessor of "abac", while "cba" is not a predecessor of "bcad".
A word chain is a sequence of words [word1, word2, ..., wordk] with k >= 1, where word1 is a predecessor of word2, word2 is a predecessor of word3, and so on. A single word is trivially a word chain with k == 1.

Return the length of the longest possible word chain with words chosen from the given list of words.
Input :   || Output :
*/
// Bruteforce -----Recursion------>
// Time Complexity: O(N*N * l)
// Reason: We are setting up two nested loops and the compare function can be estimated to l, where l is the length of the longest string in the words [ ] array. Also, we are sorting so the time complexity will be (N^2 * l + NlogN)
// Space Complexity: O(N)
// Reason: We are only using a single array of size n.
bool checkPossible(string &s1, string &s2)
{
    if (SZ(s1) != SZ(s2) + 1)
        return false;
    int first = 0, second = 0;
    while (first < SZ(s1))
    {
        if (s1[first] == s2[second])
        {
            first++;
            second++;
        }
        else
        {
            first++;
        }
    }
    if (first == SZ(s1) && second == SZ(s2))
        return true;
    return false;
}
bool static comp(string &s1, string &s2)
{
    return SZ(s1) < SZ(s2);
}

int longestStrChain(vector<string> &nums)
{
    int n = SZ(nums);
    sort(nums.begin(), nums.end(), comp);
    VI dp(n, 1);
    int maxi = 1;
    for (int i = 0; i < n; i++)
    {
        for (int prev = 0; prev < i; prev++)
        {
            if (checkPossible(nums[i], nums[prev]) && 1 + dp[prev] > dp[i])
            {
                dp[i] = 1 + dp[prev];
            }
        }
        if (dp[i] > maxi)
        {
            maxi = dp[i];
        }
    }
    return maxi;
}
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
46.
ANS :
Input :   || Output :
*/
// Bruteforce -----Recursion------>
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
47.
ANS :
Input :   || Output :
*/
// Bruteforce -----Recursion------>
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
48.
ANS :
Input :   || Output :
*/
// Bruteforce -----Recursion------>
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
49.
ANS :
Input :   || Output :
*/
// Bruteforce -----Recursion------>
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
50.
ANS :
Input :   || Output :
*/
// Bruteforce -----Recursion------>
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
51.
ANS :
Input :   || Output :
*/
// Bruteforce -----Recursion------>
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
52.
ANS :
Input :   || Output :
*/
// Bruteforce -----Recursion------>
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
53.
ANS :
Input :   || Output :
*/
// Bruteforce -----Recursion------>
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
54.
ANS :
Input :   || Output :
*/
// Bruteforce -----Recursion------>
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
55.
ANS :
Input :   || Output :
*/
// Bruteforce -----Recursion------>
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
56.
ANS :
Input :   || Output :
*/
// Bruteforce -----Recursion------>
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
57.
ANS :
Input :   || Output :
*/
// Bruteforce -----Recursion------>
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
    // cout << "All paths " << allPaths(3, 7) << endl;
    // cout << "All paths " << uniquePathsRecr(3, 2) << endl;
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

    // vector<int> wt = {1, 2, 4, 5};
    // vector<int> val = {5, 4, 8, 6};
    // cout << "0 1 Knapsack R " << knapsackR(wt, val, 5) << endl;
    // cout << "0 1 Knapsack M " << knapsackM(wt, val, 5) << endl;
    // cout << "0 1 Knapsack T " << knapsackT(wt, val, 5) << endl;
    // cout << "0 1 Knapsack S " << knapsackSO(wt, val, 5) << endl;

    // vector<int> coin = {1, 2, 3};
    // cout << "Min Coin Change R " << coinChangeR(coin, 11) << endl;
    // cout << "Min Coin Change M " << coinChangeM(coin, 11) << endl;
    // cout << "Min Coin Change T " << coinChangeT(coin, 11) << endl;
    // cout << "Min Coin Change S " << coinChangeSO(coin, 11) << endl;

    // cout << "Min Coin Change R " << coinChangeRII(coin, 11) << endl;
    // cout << "Min Coin Change M " << coinChangeMII(coin, 11) << endl;
    // cout << "Min Coin Change T " << coinChangeTII(coin, 11) << endl;
    // cout << "Min Coin Change S " << coinChangeSOII(coin, 11) << endl;

    // int val[4] = {1,4,5,7};
    // int wt[4] = {1,3,4,5};
    // int val[4] = {6, 1, 7, 7};
    // int wt[4] = {1, 3, 4, 5};
    // cout << "Max Knapsack " << knapSackRII(4, 8, val, wt) << endl;
    // cout << "Max Knapsack " << knapSackMII(4, 8, val, wt) << endl;
    // cout << "Max Knapsack " << knapSackTII(4, 8, val, wt) << endl;
    // cout << "Max Knapsack " << knapSackSOII(4, 8, val, wt) << endl;

    // VI rod = {2, 5, 7, 8, 10};
    // cout << "Rod len " << cutRodR(rod, 5) << endl;
    // cout << "Rod len " << cutRodM(rod, 5) << endl;
    // cout << "Rod len " << cutRodT(rod, 5) << endl;
    // cout << "Rod len " << cutRodSO(rod, 5) << endl;

    // string s1 = "brute";
    // string s2 = "groot";

    // cout << "The Longest Common Subsequence is ";
    // lcs(s1, s2);
    // cout << "The Longest Common Substring is " << longestCommonSubstringT(s1, s2) << endl;
    // cout << "The Longest Common Substring is " << longestCommonSubstringSO(s1, s2) << endl;
    // cout << "Longest common palindromic subsequences " << longestPalindromeSubseqSO(s1) << endl;
    // cout<<"Mini Insertion to make palindrom is "<<minInsertionSO(s1)<<endl;
    // cout << "Insert and deletion " << canYouMakeSO(s1, s2) << endl;
    // cout << "Shortes supersequence " << shortestSupersequence(s1, s2) << endl;
    // string p = "a*at";
    // string s = "chat";
    // cout << "Distinct Subsequences " << numDistinctR(s, t) << endl;
    // cout << "Distinct Subsequences " << numDistinctM(s, t) << endl;
    // cout << "Distinct Subsequences " << numDistinctT(s, t) << endl;
    // cout << "Distinct Subsequences " << numDistinctSO(s, t) << endl;
    // cout << "Edited distance is R " << editDistanceR(s, t) << endl;
    // cout << "Edited distance is M " << editDistanceM(s, t) << endl;
    // cout << "Edited distance is T " << editDistanceT(s, t) << endl;
    // cout << "Edited distance is SO " << editDistanceSO(s, t) << endl;
    // cout << "Is matching " << wildcardMatchingR(p, s) << endl;
    // cout << "Is matching " << wildcardMatchingM(p, s) << endl;
    // cout << "Is matching " << wildcardMatchingT(p, s) << endl;
    // cout << "Is matching " << wildcardMatchingSO(p, s) << endl;
    // VI prices = {1, 2, 3, 0, 2};
    // cout << "Max profit " << maxProfitI(prices) << endl;
    // cout << "Max profit " << maxProfitIIR(prices) << endl;
    // cout << "Max profit " << maxProfitIIM(prices) << endl;
    // cout << "Max profit " << maxProfitIIT(prices) << endl;
    // cout << "Max profit " << maxProfitIISO(prices) << endl;
    // cout << "Max profit " << maxProfitIIVO(prices) << endl;
    // cout << "Max profit " << maxProfitIIIR(prices) << endl;
    // cout << "Max profit M " << maxProfitIIIM(prices) << endl;
    // cout << "Max profit T " << maxProfitIIIT(prices) << endl;
    // cout << "Max profit SO " << maxProfitIIISO(prices) << endl;
    // cout << "Max profit R " << maxProfitIIItR(prices, 2) << endl;
    // cout << "Max profit M " << maxProfitIIItM(prices, 2) << endl;
    // cout << "Max profit T " << maxProfitIIItT(prices, 2) << endl;
    // cout << "Max profit T " << maxProfitIIItSO(prices, 2) << endl;
    // cout << "Max profit Cooldown R " << maxProfitCoolR(prices) << endl;
    // cout << "Max profit Cooldown M " << maxProfitCoolM(prices) << endl;
    // cout << "Max profit Cooldown T " << maxProfitCoolT(prices) << endl;
    // cout << "Max profit Cooldown SO " << maxProfitCoolSO(prices) << endl;
    // VI arr = {10, 9, 2, 5, 3, 7, 101, 18};
    // cout << "Longest inc subse.. " << lengthOfLISR(arr) << endl;
    // cout << "Longest inc subse.. " << lengthOfLISM(arr) << endl;
    // cout << "Longest inc subse.. " << lengthOfLIST(arr) << endl;
    // cout << "Longest inc subse.. " << lengthOfLISSO(arr) << endl;
    // cout << "Longest inc subse.. " << endl;
    // VI a = longestIncreasingSubsequenceTB(SZ(arr), arr);
    // VI a = largestDivisibleSubset(arr);
    // printVector(a);
    VS words = {"a", "b", "ba", "bca", "bda", "bdca"};
    cout<<"Chain length "<<longestStrChain(words);

    //  End code here-------->>

    return 0;
}
