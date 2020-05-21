import numpy as np

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]])

# number of rows and columns
matrix.shape

# number of all elements in matrix
matrix.size

# dimension
matrix.ndim

# define a function add 100
add_100 = lambda i: i + 100
# def add_100(i): return i + 100

vectorized_add_100 = np.vectorize(add_100)
vectorized_add_100(matrix)

# define a function ax + b


def ax_p_b(i): return a*i + b


a = 10
b = 1

vectorized_ax_p_b = np.vectorize(ax_p_b)
vectorized_ax_p_b(matrix)

# max, 9
np.max(matrix)
# miin, 1
np.mmin(matrix)

# find maximumm elemment in each column
np.max(matrix, axis=0)
# find maximum elmt in each row
np.max(matrix, axis=1)

# mean, variance, standard deviation
np.mean(matrix)
np.var(matrix)
np.std(matrix)

# find mean in each colummn
np.mean(matrix, axis=0)

means = np.mean(matrix, axis=0)
means[0]

# reshape matrix into m * n matrix
matrix.reshape(m, n)

matrix
matrix.reshape(1, 9)

# in reshape, -1 means as mmany as needed

# maximum # of columns as many as needed
# i.e matrix into a single row vector
matrix.reshape(1, -1)

# maximum # of rows as many as needed
# i.e. matrix into a singlee column vector
matrix.reshape(-1, 1)

# into 1D array of that length
matrix.reshape(matrix.size)

# flatten is shame as matrix.reshape(-1,1)
matrix.flatten()


# transpose matrix
matrix.T

# trnaspose vector
v = [1, 2, 3, 4, 5, 6]
v_t = np.array(v).T
v_t.shape  # (6, )

# transpopse row vector
[v]
v_r_t = np.array([v]).T
v_r_t.shape  # (6, 1)


# Rank
np.linalg.matrix_rank(matrix)  # 2, col3 = 3*(col2-col1)

np.linalg.matrix_rank([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 3


# Determinant
np.linalg.det(matrix)

np.linalg.det([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# diagonal
matrix.diagonal()

# diagonal one above the main diagonal
matrix.diagonal(offset=1)

# diagonal one below the main diagonal
matrix.diagonal(offset=-1)

# trace
matrix.trace()

sum(matrix.diagonal())


# eigenvalues, eigenvectors

eigenvalues, eigenvectors = np.linalg.eig(matrix)
eigenvalues
eigenvectors

# dot product
v_a = np.array([1, 2, 3])
v_b = np.array([4, 5, 6])

np.dot(v_a, v_b)
v_a @ v_b

# adding and subtracting matrices

matrix_a = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

matrix_b = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# add
np.add(matrix_a, matrix_b)
matrix_a + matrix_b

# subtract
np.subtract(matrix_b, matrix_a)
matrix_b - matrix_a

# multiply
np.dot(matrix_a, matrix_b)
matrix_a @ matrix_b

# elementwise
matrix_a * matrix_b

# invert a matrix
mat = np.array([
    [1, 1, 1],
    [2, 3, 5],
    [6, 0, 1]
])

np.linalg.inv(mat)
mat @ np.linalg.inv(mat)  # ideentity matrix

# anoter example, which is weird
np.linalg.inv(matrix)
matrix @ np.linalg.inv(matrix)  # not identity matrix

# it's because

np.linalg.det(mat)  # ~= 0
np.linalg.det(matrix)  # = 0
# mat is invertible, while matrix is not invertible


# generating random value
# set seed
np.random.seed(0)
# generate 3 random floats between 0.0 and 1.0
np.random.random(3)

# generate k random integers between n(inclusive) and m(exclusive)
np.random.randint(n, m, k)

np.random.randint(0, 3, 2)  # rand integers among 0, 1, 2

# draw three numbers from a normal distribution with m = 0.0, s = 1.
np.random.normal(0.0, 1.0, 3)

# three numbers from a logistic distribution with m = 0.0, scalar = 1.0
np.random.logistic(0.0, 1.0, 3)

# 3 numbers 1.0 <= x <2.0
np.random.uniform(1.0, 2.0, 3)

a = np.random.randn(2, 3)
print(a)
b = np.random.randn(2, 1)
print(b)

c = a + b
print(c)

a = np.random.randn(3, 3)
print(a)
b = [[1, 1, 2]]
c = a*b
print(c)

J = a*b + a*c - (b+c)
= (a-1
    raise NotImplementedError
