use rand::{thread_rng, Rng};
use std::vec;
use rand_distr::{Distribution, Normal};
use serde::{Serialize, Deserialize};

use crate::trainer::{HaltCondition, Trainer, LossFunction};

static DEFAULT_LEARNING_RATE: f64 = 0.3f64;
static DEFAULT_MOMENTUM: f64 = 0f64;
static DEFAULT_EPOCHS: u32 = 1000;

#[derive(Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    LeakyReLU,
    TanH
}

#[derive(Serialize, Deserialize)]
pub struct NN {
    layers: Vec<u32>,
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<f64>>,
    activation: ActivationFunction,
}

impl NN {
    pub fn run(&self, data: &Vec<f64>) -> Vec<f64> {
        let (_, outputs) = self.forward(data);
        outputs.last().unwrap().clone()
    }


    pub fn save_as_json(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json_string = serde_json::to_string_pretty(self)?;
        std::fs::write(filename, json_string)?;
        Ok(())
    }

    pub fn load_from_json(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json_string = std::fs::read_to_string(filename)?;
        let nn: NN = serde_json::from_str(&json_string)?;
        Ok(nn)
    }

    fn get_activation_function(&self) -> fn(f64) -> f64 {
        match self.activation {
            ActivationFunction::ReLU => Self::relu,
            ActivationFunction::Sigmoid => Self::sigmoid,
            ActivationFunction::LeakyReLU => Self::leaky_relu,
            ActivationFunction::TanH => Self::tanh,
        }
    }

    fn get_activation_derivative(&self) -> fn(f64) -> f64 {
        match self.activation {
            ActivationFunction::ReLU => Self::relu_derivative,
            ActivationFunction::Sigmoid => Self::sigmoid_derivative,
            ActivationFunction::LeakyReLU => Self::leaky_relu_derivative,
            ActivationFunction::TanH => Self::tanh_derivative,
        }
    }

    pub fn activation(&mut self, activation: ActivationFunction) -> &mut NN {
        
        self.activation = activation;
        self
    }
    
    fn compute_layer_input(inputs: &[f64], weights: &[Vec<f64>], biases: &[f64]) -> Vec<f64> {
        weights
            .iter()
            .zip(biases)
            .map(|(w, &b)| Self::dot_product(inputs, w).unwrap_or(0.0) - b)
            .collect()
    }

    pub(crate) fn forward(&self, data: &Vec<f64>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {

        let input = data;
        let num_layers = self.layers.len();

        let mut neuron_outputs: Vec<Vec<f64>> = Vec::with_capacity(num_layers);
        let mut neuron_inputs: Vec<Vec<f64>> = Vec::with_capacity(num_layers);
        let activation_function = self.get_activation_function();
        let initial_inputs = Self::compute_layer_input(&input, &self.weights[0], &self.biases[0]);
        let initial_outputs = initial_inputs.iter().map(|&x| activation_function(x)).collect();

        neuron_inputs.push(initial_inputs);
        neuron_outputs.push(initial_outputs);

        for layer_index in 1..num_layers - 1 {
            let previous_output = &neuron_outputs[layer_index - 1];
            let layer_inputs = Self::compute_layer_input(
                previous_output,
                &self.weights[layer_index],
                &self.biases[layer_index],
            );
            let layer_outputs = layer_inputs.iter().map(|&x| activation_function(x)).collect();

            neuron_inputs.push(layer_inputs);
            neuron_outputs.push(layer_outputs);
        }

        (neuron_inputs, neuron_outputs)
    }

    fn compute_output_delta(last_output: &Vec<f64>, outputs: &Vec<f64>, activation_derivative: fn(f64) -> f64) -> Vec<f64> {
        last_output
            .iter()
            .zip(outputs)
            .map(|(&net_output, &desired_output)| {
                (desired_output - net_output) * activation_derivative(net_output)
            })
            .collect()
    }

    fn update_weights_and_biases(
        weights: &mut Vec<Vec<f64>>,
        biases: &mut Vec<f64>,
        inputs: &Vec<f64>,
        delta: &Vec<f64>,
        learning_rate: f64,
        momentum: f64,
        prev_weights_delta: &mut Vec<Vec<f64>>,
        prev_biases_delta: &mut Vec<f64>,
    ) {
        for (i, weight_vec) in weights.iter_mut().enumerate() {
            for ((w, &input), u) in weight_vec
                .iter_mut()
                .zip(inputs)
                .zip(prev_weights_delta[i].iter_mut())
            {
                let delta_w = learning_rate * delta[i] * input + momentum * *u;
                *w += delta_w;
                *u = delta_w;
            }

            let delta_b = learning_rate * delta[i] + momentum * prev_biases_delta[i];
            biases[i] -= delta_b;
            prev_biases_delta[i] = delta_b;
        }
    }

    fn compute_hidden_delta(
        next_layer_delta: &Vec<f64>,
        cur_weights: &Vec<Vec<f64>>,
        cur_output: &Vec<f64>,
        activation_derivative: fn(f64) -> f64,
    ) -> Vec<f64> {
        cur_output
            .iter()
            .enumerate()
            .map(|(j, &output)| {
                let sum: f64 = next_layer_delta
                    .iter()
                    .enumerate()
                    .map(|(k, &delta)| delta * cur_weights[k][j])
                    .sum();
                sum * activation_derivative(output)
            })
            .collect()
    }

    pub(crate) fn backward(
        &mut self,
        data: &(Vec<f64>, Vec<f64>),
        network_data: (Vec<Vec<f64>>, Vec<Vec<f64>>),
        learning_rate: f64,
        momentum: f64,
        prev_weights_delta: &mut Vec<Vec<Vec<f64>>>,
        prev_biases_delta: &mut Vec<Vec<f64>>,
    ) {
        let (inputs, outputs) = data;
        let num_layers = self.layers.len();
        let (_, neuron_outputs) = network_data;
        let activation_derivative = self.get_activation_derivative();
        let last_output = neuron_outputs.last().unwrap();
        let mut deltas: Vec<Vec<f64>> = vec![Self::compute_output_delta(last_output, outputs, activation_derivative)];

        for index in (2..num_layers).rev() {
            let delta_index = num_layers - index - 1;

            Self::update_weights_and_biases(
                &mut self.weights[index - 1],
                &mut self.biases[index - 1],
                &neuron_outputs[index - 2],
                &deltas[delta_index],
                learning_rate,
                momentum,
                &mut prev_weights_delta[index - 1],
                &mut prev_biases_delta[index - 1],
            );

            let current_delta = Self::compute_hidden_delta(
                &deltas[delta_index],
                &self.weights[index - 1],
                &neuron_outputs[index - 2],
                activation_derivative,
            );
            deltas.push(current_delta);
        }

        Self::update_weights_and_biases(
            &mut self.weights[0],
            &mut self.biases[0],
            inputs,
            &deltas.last().unwrap(),
            learning_rate,
            momentum,
            &mut prev_weights_delta[0],
            &mut prev_biases_delta[0],
        );
    }

    pub fn new(layers: &Vec<u32>) -> NN {
        if layers.is_empty() {
            panic!("Layers cannot be empty");
        }

        let mut nn = NN {
            layers: layers.clone(),
            weights: vec![],
            biases: vec![],
            activation: ActivationFunction::Sigmoid,
        };

        // nn.generate_weights();
        nn.generate_weights_he();

        nn
    }


    fn generate_weights(&mut self) {
        let mut rng = rand::thread_rng();
        self.weights = self
            .layers
            .windows(2)
            .map(|layer_pair| {
                (0..layer_pair[1])
                    .map(|_| {
                        (0..layer_pair[0])
                            .map(|_| rng.gen_range(-1.0..1.0))
                            .collect()
                    })
                    .collect()
            })
            .collect();

        self.biases = self
            .layers
            .iter()
            .skip(1)
            .map(|&l| (0..l).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
    }

    fn generate_weights_he(&mut self) {

        let mut rng = thread_rng();
        
        self.weights = self.layers.windows(2).map(|layer_pair| {
            let fan_in = layer_pair[0] as f64;
            let he_std_dev = (2.0 / fan_in).sqrt();
            let normal_dist = Normal::new(0.0, he_std_dev).unwrap();
            
            (0..layer_pair[1])
                .map(|_| {
                    (0..layer_pair[0])
                        .map(|_| normal_dist.sample(&mut rng))
                        .collect()
                })
                .collect()
        }).collect();
    
        let normal_dist = Normal::new(0.0, 1.0).unwrap();
        self.biases = self.layers.iter().skip(1).map(|&l| {
            (0..l).map(|_| normal_dist.sample(&mut rng)).collect()
        }).collect();
    }

    pub fn train<'a>(&'a mut self, examples: &'a [(Vec<f64>, Vec<f64>)]) -> Trainer {
        let output_layer_size = *self.layers.last().unwrap() as usize;
        for example in examples {
            let (input, output) = example;
            if input.len() != self.layers[0] as usize {
                panic!("Input vector must have the same length as input layer.");
            }
            if output.len() != output_layer_size {
                panic!("Output vector must have the same length as output layer.");
            }
        }

        Trainer {
            examples: &examples,
            momentum: DEFAULT_MOMENTUM,
            prev_weights_delta: self
                .weights
                .iter()
                .map(|layer| vec![vec![0.0; layer[0].len()]; layer.len()])
                .collect(),
            prev_biases_delta: self
                .biases
                .iter()
                .map(|layer| vec![0.0; layer.len()])
                .collect(),
            learning_rate: DEFAULT_LEARNING_RATE,
            halt_condition: HaltCondition::Epochs(DEFAULT_EPOCHS),
            log_interval: None,
            nn: self,
            loss_function: LossFunction::MeanSquaredError,
        }
    }



    fn dot_product(first: &[f64], second: &[f64]) -> Result<f64, &'static str> {
        if first.len() != second.len() {
            return Err("Must have the same number of elements");
        }

        let result = first.iter().zip(second).map(|(x, y)| x * y).sum();
        Ok(result)
    }

    fn sigmoid(y: f64) -> f64 {
        1f64 / (1f64 + (-y).exp())
    }

    fn sigmoid_derivative(y: f64) -> f64 {
        y * (1f64 - y)
    }

    fn relu(x: f64) -> f64 {
        x.max(0.0)
    }
    
    fn relu_derivative(x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }

    fn leaky_relu(x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            0.01 * x
        }
    }

    fn leaky_relu_derivative(x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.01 }
    }

    fn tanh(x: f64) -> f64 {
        x.tanh()
    }

    fn tanh_derivative(x: f64) -> f64 {
        1.0 - x.powi(2)
    }
}