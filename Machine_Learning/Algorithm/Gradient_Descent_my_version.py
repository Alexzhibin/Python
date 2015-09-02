#Part I Create loss/cost function 
from sympy import *
import numpy as np
from numpy import *
x = Symbol('x')
y = Symbol('y')
m = Symbol('m')
b = Symbol('b')
random.seed(10)
points = random.random((50,2))
# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / (2*len(points))

cost_b = compute_error_for_line_given_points(b,m,points)
print cost_b

#####Devirative
##The purpose of doing this is find out the devirative for the polynon, no matter how many variable there are
#from sympy import *
#import numpy as np
#x = Symbol('x')
#y = Symbol('y')
#m = Symbol('m')
#b = Symbol('b')
#devirative_b = diff((-b - m*x + y)**2/2,b)
#devirative_m = diff((-b - m*x + y)**2/2,m)
#devirative_b = b + m*x - y
#devirative_m = -x*(-b - m*x + y)
##print devirative_b
#print devirative_m

#Make the Devirative minimum equal to zero 
learningrate=0.0001
current_m = 0.1
current_b= 0.1
start_x = points[0,0]
start_y = points[0,1]
num_iterations = 1000
#
import pandas as pd
df4 = pd.DataFrame({'num.iterations':(),'cost':()})
#
for k in range(num_iterations):
    for i in range(len(points)):
         if current_m==current_b==0:
                    print "break"
                    break
         else:       
            start_x   =  points[i,0]
            start_y   =  points[i,1]
            devirative_b = current_b + current_m*start_x - start_y
            devirative_m = -start_x*(-current_b - current_m*start_x + start_y)
            current_m -= learningrate*devirative_m
            current_b -= learningrate*devirative_b

    df4.loc[k] = [compute_error_for_line_given_points(current_b,current_m,points),k] 

#print df4
#plt.plot(df4)

import pylab as pl
import numpy as np
x=df4['num.iterations']
y=df4['cost']
pl.xlabel('num.iterations')
pl.ylabel('cost')
pl.plot(x, y)
pl.show()
