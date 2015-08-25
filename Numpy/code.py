1. Basis 
from numpy import *
a = arange(15).reshape(3,5)
a
a.shape #shape: 3 rows, 5 colons 
a.ndim  #numbers of dimensions
a.dtype.name #64 bit
a.itemsize #the size in bytes of each element of the array.an array of elements of type float64 has itemsize 8 (=64/8), while one of type complex32 has itemsize 4 (=32/8). 
a.size  #number of observations
type(a) #type : array 
b = array([6,7,8])
b
type(b) #as same as a

2. Create and Printing
          from numpy import *
###Dimension
#One dimensional 
#a = array(1,2,3,4) #Wrong
a = array([1,2,3,4])  #Right
#Two dimensional 
b =  array([(1,2,3,4),(5,6,7,8)])
b
#Three dimensional 
c = array([(1,4,3,2),(6,5,4,2),(5,6,4,2)])
c
#type specified
c = array([(1,4,3,2),(6,5,4,2),(5,6,4,2)] , dtype=complex)
c
#some function for creating the array
zeros((3,4))  #create 3 dimensions of zero, and 4 zero in each dimension
ones((2,3,4),dtype=int16) #create 2 set , 3 dimensions and 4 zeros in each dimension
empty((2,3)) #creates an array whose initial content is random and depends on the state of the memory. By default, the dtype of the created array is float64
###Creations 
#Create sequences of numbers
arange(10,30,5) #from 10 to 30(not inclued) , step is 5
arange(0,2,0.3) #from 0 to 2(not inclued) , step is 0.3
#create sequence of numbers with certain numbers
linspace(0,2,9) # 9 numbers from 0 to 2(included)
x = linspace(0, 2*pi,100)  # useful to evaluate function at lots of points
f = sin(x)
###Printing Arrays
a = arange(6)  #1d array
print a
b = arange(12).reshape(4,3)  #2d array
print b
c = arange(24).reshape(2,3,4)  #3d array
print c

3. Basic Operation 
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

4. Indexing, slicing and Iterating
