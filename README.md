# torch-extension
Skeleton code for custom torch extension.

# Problem statement.

Given a tensor A of shape (s1,s2), and a list of indexes B of shape (s1,s3), we need to produce
function gather-reduce such that tensor C of shape(s1,s2) such that
C[i][j] = sum(A[b[i][k]][j]) for k (0,s2)
This is a form of neighbourhood aggregation.


Todo List:

1. Implement this naively in python. Do a sanity check if something better exists. (DONE)
2. Implement naively in cpp (DONE)
3. Implement with multi threads
4. Implement with cuda.
5. Write both a forward and backward function.

# Future work.

1. Add distributed torch extension.
2. Add distributed gpu extension.
3. Support autograd function.
4. Study the memory management in python. 
