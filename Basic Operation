###Arithmetic Operations
a = array([20,30,40,50])
b = arange(4)
b
c = a - b
c
b ** 2
10 * sin(a)
a < 35

###element or matrix operation 
A = array([[1,1],[0,1]])
B = array([[2,0],[3,4]])
A*B  #elementwise product  element*element
dot(A,B)  #matrix product  matrix * matrix

###+= and *=
a= ones((2,3),dtype=int)
b = random.random((2,3))
a *= 3
b /=a 
a += b
a

###operating with arrays of different types
a = ones(3,dtype=int32)
b = linspace(0,pi,3)
b.dtype.name
c = a+b
c
c.dtype.name
d = exp(c*1j)
d
d.dtype.name

###ndarray class
a = random.random((2,3))
a
a.sum()
a.min()
a.max()

###Axis
b = arange(12).reshape(3,4)
b
b.sum(axis=0)    #sum of each column
b.min(axis=1)    #min of each row
b.cumsum(axis=1)  #cumulative sum along each row

###Universal Functions, such as sin, cos, and exp
b = arange(3)
b
exp(b)
sqrt(b)
c = array([2.,-1.,4.])
add(b,c)
