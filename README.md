![octologo_rusty_net_1730434792](https://github.com/user-attachments/assets/ced8f0fd-3d32-4fd0-9dbf-d5621d58a135)

# Overview
rusty_net is a user friendly, easy to use Rust library for building [feedforward neural networks](https://en.wikipedia.org/wiki/Feedforward_neural_network). It generates fully-connected multi-layer artificial neural networks that are trained via [backpropagation](https://en.wikipedia.org/wiki/Backpropagation).

The library allows for easy configuration of key parameters, including momentum, learning rate, and halt conditions, simplifying the training process.

```rust
let mut nn = NN::new(&vec![3,10, 1]); // Network can be defined via a vec specifying the layers.
nn.train(&examples) 
    .rate(0.1) // Adjusting learning rate
    .momentum(0.9) // Adjusting momentum
    .log_interval(Some(100)) // Adjusting log interval (print eror each 100 epochs)
    .halt_condition(Epochs(5000)) // Adjusting halt condition (network will stop after 5000 epochs)
    .go();
```

# AND example

This example sets up a neural network with two nodes in the input layer, one hidden layer consisting of five nodes, and a single node in the output layer. The network is trained using examples of the AND function. The training only starts after the `.go()` method is called. 

```rust
use rusty_net::{NN, HaltCondition::Epochs};

// create examples of the AND function
// the network is trained on tuples of vectors where the first vector
// is the inputs and the second vector is the expected outputs
let examples = [
    (vec![0f64, 0f64], vec![0f64]),  // 0 AND 0 = 0
    (vec![0f64, 1f64], vec![0f64]),  // 0 AND 1 = 0
    (vec![1f64, 0f64], vec![0f64]),  // 1 AND 0 = 0
    (vec![1f64, 1f64], vec![1f64]),  // 1 AND 1 = 1
];

let mut net = NN::new(&vec![2, 5, 1]);
    
net.train(&examples)
    .halt_condition(HaltCondition::Epochs(10000))
    .log_interval(Some(100))
    .momentum(0.1)
    .rate(0.3)
    .go();
    
// evaluate the network to see if it learned the AND function
for &(ref inputs, ref outputs) in examples {
    let results = net.run(inputs);
    let (result, key) = (results[0].round(), outputs[0]);
    assert!(result == key);
}
