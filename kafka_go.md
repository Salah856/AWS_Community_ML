
# Dealing with Kafka using Golang 

In this article, you will learn how to write and read JSON records using Kafka. Its name was inspired by the author Franz Kafka, because Kafka software is optimized for writing. Kafka is written in Scala and Java. It was originally developed by LinkedIn and donated to the Apache Software Foundation back in 2011. Kafka's design was influenced by transaction logs.

The main advantage of Kafka is that it can be used to store lots of data fast, which might interest you when you have to work with huge amounts of real-time data. However, its disadvantage is that in order to maintain that speed, the data is read-only and stored in a naive way.

## Write to Kafka 

The following program, which is named writeKafka.go , will illustrate how to write data to Kafka. In Kafka terminology, writeKafka.go is a producer. The utility will be presented in three parts.

The first part of writeKafka.go is as follows:

```go 

package main

import (
 "context"
 "encoding/json"
 "fmt"
 "github.com/segmentio/kafka-go"
 "math/rand"
 "os"
 "strconv"
 "time"
)


func main() {
 MIN := 0
 MAX := 0
 TOTAL := 0
 topic := ""

 if len(os.Args) > 4 {
  MIN, _ = strconv.Atoi(os.Args[1])
  MAX, _ = strconv.Atoi(os.Args[2])
  TOTAL, _ = strconv.Atoi(os.Args[3])
  topic = os.Args[4]
 } else {

  fmt.Println("Usage:", os.Args[0], "MIX MAX TOTAL TOPIC")
   return
 }


```


The second part of writeKafka.go contains the following Go code:

```go 

type Record struct {
  Name string `json:"name"`
  Random int `json:"random"`
}

func random(min, max int) int {
  return rand.Intn(max-min) + min
}
```

The Record structure is used to store the data that will be written to the desired Kafka topic. The third part of writeKafka.go contains the following Go code:

```go 

partition := 0

conn, err := kafka.DialLeader(context.Background(), "tcp", "localhost:9092", topic, partition)

if err != nil {

 fmt.Printf("%s\n", err)
  return
}

rand.Seed(time.Now().Unix())


for i := 0; i < TOTAL; i++ {
 
 myrand := random(MIN, MAX)
 temp := Record{strconv.Itoa(i), myrand}
 recordJSON, _ := json.Marshal(temp)
 
 conn.SetWriteDeadline(time.Now().Add(1 * time.Second))
 conn.WriteMessages( kafka.Message{Value: []byte(recordJSON)},)
 
 if i%50 == 0 {
  fmt.Print(".")
 }
  time.Sleep(10 * time.Millisecond)
 }

 fmt.Println()
 conn.Close()

}


```

## Read from Kafka 


The following Go program, which is named readKafka.go , will illustrate how to write data to Kafka. Programs that read data from Kafka are called consumers in Kafka terminology.

```go 


package main

import (
 "context"
 "encoding/json"
 "fmt"
 "github.com/segmentio/kafka-go"
 "os"
)


type Record struct {
 Name string `json:"name"`
 Random int `json:"random"`
}


func main() {
 if len(os.Args) < 2 {
  fmt.Println("Need a Kafka topic name.")
  return
 }

 partition := 0
 topic := os.Args[1]
 fmt.Println("Kafka topic:", topic


 r := kafka.NewReader(kafka.ReaderConfig{ Brokers: []string{"localhost:9092"}, Topic: topic, Partition: partition, MinBytes: 10e3, MaxBytes: 10e6, })
 
 r.SetOffset(0)
 
 for {
   m, err := r.ReadMessage(context.Background())
   
   if err != nil {
    break
   }
 fmt.Printf("message at offset %d: %s = %s\n", m.Offset,string(m.Key), string(m.Value))
 
 temp := Record{}
 err = json.Unmarshal(m.Value, &temp)
 
 if err != nil {
  fmt.Println(err)
 }
 fmt.Printf("%T\n", temp)

 }
 r.Close()
}



```
