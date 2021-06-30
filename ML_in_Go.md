
# ML in Go 

### Calculating simple statistical properties

Statistical learning is a branch of applied statistics that is related to machine learning.

Machine learning, which is closely related to computational statistics, is an area of computer science that tries to learn from data and make predictions about it without being specifically programmed to do so. 


In this article, you are going to learn how to calculate basic statistical properties such as the mean value, the minimum and the maximum values of your sample, the median value, and the variance of the sample. These values give you a good overview of your sample without going into too much detail. However, generic values that try to describe your sample can easily trick you by making you believe that you know your sample well without this being true.

All these statistical properties will be computed in stats.go , which will be presented in five parts. Each line of the input file contains a single number, which means that the input file is read line by line. Invalid input will be ignored without any warning messages.

Notice that input will be stored in a slice in order to use a separate function for calculating each property. Also, as you will see shortly, the values of the slice will be sorted before processing them.


```go 

package main

import (
  "bufio"
  "flag"
  "fmt"
  "io"
  "math"
  "os"
  "sort"
  "strconv"
  "strings"
)


func min(x []float64) float64 {
  return x[0]
}

func max(x []float64) float64 {
  return x[len(x)-1]
}


func meanValue(x []float64) float64 {
 sum := float64(0)
 for _, v := range x {
   sum = sum + v
 }
  return sum / float64(len(x))
}


func medianValue(x []float64) float64 {

 length := len(x)
 if length%2 == 1 {
 // Odd
  return x[(length-1)/2]
 } else {
  // Even
   return (x[length/2] + x[(length/2)-1]) / 2
 }
  return 0
}


```
