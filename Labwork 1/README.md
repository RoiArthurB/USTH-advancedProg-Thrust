# Labwork 1 – MAP, GATHER and SCATTER

For each exercise, check its computation time (launch the calculation two or three times in a loop before to measure the time, since GPU takes some time to wake-up).
Some pieces of implementation are provided in skeletons ... use them!

## Exercise 1: Just try the previous examples, by doing it yourself ...

1. For two vectors containing data on the host (with at least 2 16 values).
2. For two vectors of the same size, but using counting and constant iterators.
3. For three vectors (on CPU) of the same size.

## Exercise 2: Separate the odd and the even number ...

1. Write a function that takes as input a large vector of integers, and that separates and returns the same vector containing first the data at even indexes and then the ones at odd indexes. This first function uses *GATHER*.
e.g.: {1, 2, 3, 4, 5, 6} becomes {1, 3, 5, 2, 4, 6}.
2. Do the same using *SCATTER*.
3. Do the same but for more heavy objects (records containing some useless data). You may use *SCATTER* or *GATHER*, as you prefer.