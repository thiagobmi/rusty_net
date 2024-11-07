![octologo_rusty_net_1730434967](https://github.com/user-attachments/assets/b7f1bba4-f3e0-4a6a-9272-e4450f58bf47)


# Overview
rusty_net is a user-friendly Rust library for building feedforward neural networks. It generates fully connected, multi-layer neural networks that are trained using backpropagation.

The library allows for easy configuration of key parameters, including momentum, learning rate, and halt conditions, simplifying the training process.

```rust
// Creating a network with:
// 2 nodes in input layer,
// two hidden layers, with 3 and 5 nodes,
// and an output layer with 2 nodes.
let mut nn = NN::new(&vec![2, 3, 5, 2]); 

nn.train(&examples) 
        .rate(0.1) // Configuring learning rate
        .momentum(0.9) // Configuring momentum
        .log_interval(Some(100)) // Network will log the error each 100 epochs
        .halt_condition(Epochs(5000)) // Setting halt condition (stop after 5000 epochs)
        .go(); // Starts the training


```
![image](https://github.com/user-attachments/assets/cc48f228-71e1-4b70-8ff0-5f717bd3c6b3)

## Additional Parameters

This library offers advanced options for configuring your neural network, such as activation functions and loss functions. These parameters provide flexibility to tailor your model to specific requirements.

```rust

// Set the activation function
let mut nn = NN::new(&vec![2, 2, 1]);
nn.activation(rusty_net::ActivationFunction::LeakyReLU);

// Set the loss function
nn.train(&examples)
        .loss_function(rusty_net::LossFunction::CrossEntropy)
        .go();
```

***Activation functions:***
 - Sigmoid
 - ReLU
 - Leaky ReLU
 - TanH

***Loss functions:***
 - Mean Squared Error (MSE)
 - Cross Entropy


## Saving

You can save the weights and biases of your network in a json file by simply using:

```rust
nn.save_as_json("nn.json");
```

The data can then be loaded with:

```rust
let nn = NN::load_from_json("nn.json");
```


# AND example
This example initializes a neural network with an input layer of 2 nodes, a single hidden layer with 3 nodes, and an output layer containing 1 node. The network is trained on examples of the AND function. After calling train(&examples), additional methods are used to configure training options, though these are optional. Training begins when the go() method is called, prompting the network to learn from the provided examples.

```rust
use rusty_net::{NN, HaltCondition};

// create examples of the AND function
// the network is trained on tuples of vectors where the first vector
// is the inputs and the second vector is the expected outputs
let examples = [
    (vec![0f64, 0f64], vec![0f64]), // 0 AND 0 = 0
    (vec![0f64, 1f64], vec![0f64]), // 0 AND 1 = 0
    (vec![1f64, 0f64], vec![0f64]), // 1 AND 0 = 0
    (vec![1f64, 1f64], vec![1f64]), // 1 AND 1 = 1
];

// create a new neural network by passing a pointer to a vector
// that specifies the number of layers and the number of nodes in each layer
let mut net = NN::new(&vec![2, 3, 1]);

// train the network on the examples of the AND function
net.train(&examples)
    .halt_condition(HaltCondition::Epochs(1000))
    .log_interval(Some(100))
    .momentum(0.1)
    .rate(0.3)
    .go();

// evaluate the network to see if it learned the AND function
for &(ref inputs, ref outputs) in examples.iter() {
    let results = net.run(inputs);
    let (result, key) = (results[0].round(), outputs[0]);
    assert!(result == key);
}
```
