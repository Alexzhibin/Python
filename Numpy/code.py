1.# Basis 
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

2. #Create and Printing
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

3.# Basic Operation 
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

4.# Indexing, slicing and Iterating
#One-dimensional arrays can be indexed,sliced and iterated over
from numpy import *
a = arange(10)**3
a
a[2]
a[2:5]
a[:6:2]=-1000 # equivalent to a[0:6:2] = -1000; from start to position 6, exclusive, set every 2nd element to -1000
a
a[::-1] # reversed a
for i in a:
    print i**(1/3.),
#Multidimensional 
def f(x,y):
    return 10*x+y
b=fromfunction(f,(5,4),dtype=int)  #for dimension 1, x=0,for colomn 1 y=0,(1,1);for dimension 2,x=1,y=0,(2,1)
b
b[2,3]
b[0:5,1]   #each row in the second column of b, like matrix 
b[ :,1]    # equivalent to the previous example
b[1:3, :]  # each column in the second and third row of b
b[-1]      # the last row. Equivalent to b[-1,:]
c = array([[[0, 1,  2],
            [10,12,13]],       # a 3D array (two stacked 2D arrays)
            [[100,101,102],
            [110,112,113]]])
c.shape
 
print c[1,:,:]      # same as c[1,:,:] or c[1],  print the rest except for the first set
print c[:,:,2]      # same as c[:,:,2]

#Iterating  over multidimensional arrays is done with respect to the first axis
for row in b:
    print row 
    
#Iterating each elements
for element in b.flat:
    print element,

5. #Shape Manipulation 
#Changing the shape of an array
a = floor(10*random.random((3,4)))
a
a.shape
a.ravel()  #flatten the array
a.shape = (6,2)
a.transpose()  #transpose the array
a
a.resize((2,6)) #rebuild the array into 2 dimensions and 6 columns
a
a.reshape(3,-1)  #reshape into 3 dimensions and reduce 1 column

6. #Stacking together different arrays
a = floor(10*random.random((2,2))) 
#create random number less than 9 into 3 and 4 array in stack 
b = floor(10*random.random((2,2)))
vstack((a,b))
hstack((a,b))
column_stack((a,b))  #with 2D array
a = array([4.,2.])
b = array([2.,8.])
a[:,newaxis]    # This allows to have a 2D columns vector
column_stack((a[:,newaxis],b[:,newaxis]))
vstack((a[:,newaxis],b[:,newaxis]))   # The behavior of vstack is different

7. #Splitting one array into several smaller ones
a = floor(10 * random.random((2,12)))
a
hsplit(a,3)  #Split a into 3
hsplit(a,(3,4))   # Split a after the third and the fourth column

8. #No Copy at All
a = arange(12)
b = a
b is a 
b.shape = 3,4   # changes the shape of a
a.shape
def f(x):
    print id(x)
id(a)    #id is a unique identifier of an object
f(a)     #the same as id

8. #View or Shallow Copy
c = a.view()
c is a 
c.base is a    #c is a view of the data owned by a
c.flags.owndata
c.shape =2,6       # a's shape doesn't change
a.shape   
c[0,4] = 1234  # a's data changes
a

s = a[ : ,1:3] # spaces added for clarity; could also be written "s = a[:,1:3]"
s
s[:]=10        # s[:] is a view of s. Note the difference between s=10 and s[:]=10
a

9. #Deep Copy
d = a.copy() # a new array object with new data is created
d is a 
d.base is a #d doesn't share anything with a
d[0,0] =9999
a            #d and a is irrelevant 

10. ##Matrix
from numpy import *
A = matrix ('1.0 2.0; 3.0 4.0')
print A
type(A)

A.T  #transpose

X = matrix('5.0 7.0')
Y = X.T
Y
print A*Y #matrix multiplication
print A.I #Inverse

11.##Histograms
import numpy
import pylab
# Build a vector of 10000 normal deviates with variance 0.5^2 and mean 2
mu, sigma = 2, 0.5
v = numpy.random.normal(mu,sigma,10000)
# Plot a normalized histogram with 50 bins
pylab.hist(v, bins=50, normed=1)       # matplotlib version (plot)
pylab.show()
# Compute the histogram with numpy and then plot it
(n, bins) = numpy.histogram(v, bins=50, normed=True)  # NumPy version (no plot)
pylab.plot(.5*(bins[1:]+bins[:-1]), n)
pylab.show()
