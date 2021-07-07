# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
'''barchart'''

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import statistics
from numpy import median
import scipy.stats as stats
import pylab as pl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import iqr
import matplotlib.pyplot as plt
import scipy, scipy.stats
from scipy.stats import binom
from matplotlib import pyplot as plt
import statistics
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
# Fixing random state for reproducibility
np.random.seed(19680801)
# bar-chart 
def drawingbarchart (aList,bList): 
 # objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
  y_pos = np.arange(len(aList))
 #performance = [10,8,6,4,2,1]
 
  plt.bar(y_pos, bList, align='center', alpha=0.5)
  plt.xticks(y_pos, aList)
  plt.show()
  
  #drawingbarchart(('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp'),
   #               [10,8,6,4,2,1])

#histogram
#############################################################

#median
def getMedian(alist):
 print(median(alist))
 
#getMedian([1,3,4,3])

#############################################################

def drawinghistogram(N_points ,n_bins):
    # Generate a normal distribution, center at x=0 and y=5
    x = np.random.randn(N_points)
    y = .4 * x + np.random.randn(N_points) + 5
    
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    
    # We can set the number of bins with the `bins` kwarg
    axs[0].hist(x, bins=n_bins)
    axs[1].hist(y, bins=n_bins)
    plt.show()

#############################################################
##zscore 

def zScore(aList):
 from scipy import stats 
 print (stats.zscore(aList))  

#############################################################
#correlation

def correlation(aList,bList):
 import matplotlib
 import matplotlib.pyplot as plt
 matplotlib.style.use('ggplot')
 plt.scatter(aList,bList)
 plt.show()
 
 #############################################################
#normal dist

def normalDist(aList):
 h = sorted(aList)  #sorted
 fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed
 pl.plot(h,fit,'-o')
 pl.hist(h,density=True)      #use this to draw histogram of your data
 pl.show()            
#normalDist([1,2,3,4])


#############################################################
#pychart

def drawingPychart(labels,sizes):
 colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','red','gray']
 patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
 plt.legend(patches, labels, loc="best")
 plt.axis('equal')
 plt.tight_layout()
 plt.show()
#drawingPychart(['a','b','c'],[18,12,6])

#############################################################
#regression

def regression(x,y):
 fit = np.polyfit(x,y,1)
 fit_fn = np.poly1d(fit) 
 plt.plot(x,y, 'yo', x, fit_fn(x), '--k')
 plt.xlim(0, 10)
 plt.ylim(0, 20)
 plt.show()
#regression([1,2,3,4],[2,4,6,8])

#############################################################
#box-plot

def drawingBoxPlot(x1):
 fig = plt.figure(figsize=(8,6))
 plt.boxplot([x for x in [x1]], 0, 'rs', 1)
 plt.xticks([y+1 for y in range(len([x1]))], ['x1'])
 plt.xlabel('measurement x')
 t = plt.title('Box plot')
 plt.show()
#drawingBoxPlot([1,2,3,4,5,6,7,8,9,10,11,15,18])

#############################################################
#dot-plot

def drawDotPlot(x,y):
    plt.scatter(x,y, s=10)
    plt.show()
#drawDotPlot([1,2,3,4,5,6,7],[2,4,6,8,10,12,14])

#############################################################
#mode

def getMode(List):   
 print(statistics.mode(List))
#getMode([1,2,1,3,1,4,1])


#############################################################
#mean

def getMean(List):   
 print(statistics.mean(List))
#getMean([1,2,1,3,1,4,1])

#############################################################
#stddev

def getStandardDev(List):   
 print(statistics.stdev(List))
#getStandardDev([1,2,1,3,1,4,1])

#############################################################
#variance

def getVariance(List):   
 print(statistics.variance(List))
#getVariance([1,2,1,3,1,4,1])


###############################################################


ok = '1'

while ok  == '1' :     
    request = None 
    list_a = None 
    list_b = None 
    
    print ("Enter number 1 for drawing graphs :\n" +
           "Enter  number 2 for doing methods: ")
    
    request = input("Enter the number :  ")
    
    if request == '1' :
        print("if you want to :\n" + 
              "Draw bar-chart ( 1 ) : \n" +
              "Draw pie-chart ( 2 )  : \n" + 
              "Draw Histogram ( 3 ) :  \n"+
              "Draw correlation ( 4 ) : \n"+
              "Draw regression ( 5 ) : \n" +
              "Draw Box-plot ( 6 ) : \n " +
              "Draw Dot-plot ( 7 ) : \n " +
              "Draw Normal dist. ( 8 ) : \n " )
        
        request = input("Enter the number of choice : ")
        if request == '1' :
                list_a = input ("Enter x-axis : ") 
                list_b = input("Enter y-axis : ")
                drawingbarchart(tuple(list_a.split(',')),list_b.split(','))
                # (1,2,3,4 , 4,5,6,7) test case 
        elif request == '2' :
                list_a = input ("Enter the elements : ") 
                list_b = input("Enter the values : ")
                drawingPychart(list_a.split(),list_b.split())
        elif request == '3' : 
                drawinghistogram(int(input ("Enter Number of points  : ")),int(input ("n_bins : ")))
                   #(500,10) test case              
        elif request == '4' : 
                list_a = input ("Enter x-axis : ") 
                list_b = input("Enter y-axis : ")
                correlation(list_a.split(),list_b.split())
        
        elif request == '5' : 
                list_a = input ("Enter x-axis : ") 
                list_b = input("Enter y-axis : ")
                list_a = list_a.split(',')
                list_b = list_b.split(',')                
                regression(list(map(int, list_a)),list(map(int, list_b)))
                
        elif request == '6' : 
                list_a = input ("Enter list of Data :  ") 
                list_a= list_a.split(',')
                drawingBoxPlot(list(map(int, list_a)))   
        
        elif request == '7' : 
                list_a = input ("Enter x-axis : ") 
                list_b = input("Enter y-axis : ")
                list_a = list_a.split(',')
                list_b = list_b.split(',')
                drawDotPlot(list(map(int, list_a)),list(map(int, list_b)))
        elif request == '8' :
                 list_a = input ("Enter the list of elements : ") 
                 list_a = list_a.split(',')
                 normalDist(list(map(int, list_a)))
        else : 
            print ("the number is not correct ")
        
        ok = input("Do you want to continue :(0 , 1) ")
    elif request=="2":
         print("if you want to :\n" + 
              "get mean for some values ( 1 ) : \n" +
              "get median for some values ( 2 )  : \n" + 
              "get standard deviation for some values ( 3 ) :  \n"+
              "get variance for some values ( 4 ) : \n"+
              "get mode for some values( 5 ) : \n" +
              "get Z-score for value ( 6 ) : \n " )
         
         request = input("Enter the number of choice : ")
         if request == '1' :
              list_a = input ("Enter the values :") 
              list_a = list_a.split(',')
              getMean(list(map(int, list_a)))
         elif request == '2' :
              list_a = input ("Enter the values :") 
              list_a = list_a.split(',')
              getMedian(list(map(int, list_a)))
         elif request == '3' :
              list_a = input ("Enter the values :") 
              list_a = list_a.split(',')
              getStandardDev(list(map(int, list_a)))
         elif request == '4' :
              list_a = input ("Enter the values :") 
              list_a = list_a.split(',')
              getVariance(list(map(int, list_a)))
         elif request == '5' :
              list_a = input ("Enter the values :") 
              list_a = list_a.split(',')
              getMode(list(map(int, list_a)))
         elif request == '6' :
              list_a = input ("Enter the values :") 
              list_a = list_a.split(',')
              zScore(list(map(int, list_a)))    











