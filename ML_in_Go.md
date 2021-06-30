
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


func variance(x []float64) float64 {
 
 mean := meanValue(x)
 sum := float64(0)
 
 for _, v := range x {
    sum = sum + (v-mean)*(v-mean)
 }
  return sum / float64(len(x))
}


func main() {

  flag.Parse()

  if len(flag.Args()) == 0 {

  fmt.Printf("usage: stats filename\n")
   return
 }

  data := make([]float64, 0)
  file := flag.Args()[0]

 f, err := os.Open(file)
  if err != nil {
   fmt.Println(err)
     return
 }

defer f.Close()

```


## Regression
Regression is a statistical method for calculating relationships among variables. This section will implement linear regression, which is the most popular and simplest regression technique and a very good way to understand your data. Note that regression techniques are not 100% accurate, even if you use higher-order (nonlinear) polynomials. The key with regression, as with most machine learning techniques, is to find a good enough technique and not the perfect technique and model.


### Linear regression

The idea behind linear regression is simple: you are trying to model your data using a first-degree equation. A first-degree equation can be represented as y = a x + b .

There are many methods that allow you to find out that first-degree equation that will model your data – all techniques calculate a and b .

### Implementing linear regression

The Go code of this section will be saved in regression.go , which is going to be presented in three parts. The output of the program will be two floating-point numbers that define a and b in the first-degree equation.

The first part of regression.go contains the following code:


```go 

package main


import (
  "encoding/csv"
  "flag"
  "fmt"
  "gonum.org/v1/gonum/stat"
  "os"
  "strconv"
)

type xy struct {
  x []float64
  y []float64
}


func main() {
  
  flag.Parse()
  
  if len(flag.Args()) == 0 {
    fmt.Printf("usage: regression filename\n")
    return
  }

  filename := flag.Args()[0]

  file, err := os.Open(filename)

 if err != nil {

    fmt.Println(err)
   return

 }

defer file.Close()

r := csv.NewReader(file)

records, err := r.ReadAll()

if err != nil {
  fmt.Println(err)
  return
}

size := len(records)

data := xy{
 x: make([]float64, size),
 y: make([]float64, size),
}

for i, v := range records {
if len(v) != 2 {
fmt.Println("Expected two elements")
continue
}
if s, err := strconv.ParseFloat(v[0], 64); err == nil {
data.y[i] = s
}
if s, err := strconv.ParseFloat(v[1], 64); err == nil {
data.x[i] = s

  }
}

b, a := stat.LinearRegression(data.x, data.y, nil, false)

fmt.Printf("%.4v x + %.4v\n", a, b)
fmt.Printf("a = %.4v b = %.4v\n", a, b)

}

```

## Classification

In statistics and machine learning, classification is the process of putting elements into existing sets that are called categories. In machine learning, classification is considered a supervised learning technique, which is where a set that is considered to contain correctly identified observations is used for training before working with the actual data.


A very popular and easy-to-implement classification method is called k-nearest neighbors (k-NN). The idea behind k-NN is that we can classify data items based on their similarity with other items. The k in k-NN denotes the number of neighbors that are going to be included in the decision, which means that k is a positive integer that is usually pretty small.


The input of the algorithm consists of the k-closest training examples in the feature space. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class that is the most common among its k-NN. If the value of k is 1 , then the element is simply assigned to the class that is the nearest neighbor according to the distance metric used. The distance metric depends on the data you are dealing with. As an example, you will need a different distance metric when working with complex numbers and another when working with points in three-dimensional space.


```go 

package main


import (
 
 "flag"
 "fmt"
 "strconv"
 "github.com/sjwhitworth/golearn/base"
 "github.com/sjwhitworth/golearn/evaluation"
 "github.com/sjwhitworth/golearn/knn"
 
)


func main() {
 flag.Parse()
 if len(flag.Args()) < 2 {
   fmt.Printf("usage: classify filename k\n")
   return
 }

dataset := flag.Args()[0]

rawData, err := base.ParseCSVToInstances(dataset, false)

 if err != nil {
  fmt.Println(err)
   return
 }

k, err := strconv.Atoi(flag.Args()[1])

 if err != nil {
  fmt.Println(err)
    return
 }

cls := knn.NewKnnClassifier("euclidean", "linear", k)


```
