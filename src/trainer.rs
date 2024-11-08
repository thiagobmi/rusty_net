use serde::{Deserialize, Serialize};
use std::f64::EPSILON;
use std::time::Instant;
use crate::nn::NN;

#[derive(Serialize, Deserialize)]
pub enum HaltCondition {
    Epochs(u32),
    MSE(f64),
    Time(u32),
}

#[derive(Serialize, Deserialize)]
pub enum LossFunction {
    MeanSquaredError,
    CrossEntropy,
}

pub struct Trainer<'a> {
    pub(crate) examples: &'a [(Vec<f64>, Vec<f64>)],
    pub(crate) learning_rate: f64,
    pub(crate) momentum: f64,
    pub(crate) halt_condition: HaltCondition,
    pub(crate) log_interval: Option<u32>,
    pub(crate) prev_weights_delta: Vec<Vec<Vec<f64>>>,
    pub(crate) prev_biases_delta: Vec<Vec<f64>>,
    pub(crate) loss_function: LossFunction,
    pub(crate) nn: &'a mut NN,
}

impl<'a> Trainer<'a> {
    pub fn halt_condition(&mut self, halt_condition: HaltCondition) -> &mut Trainer<'a> {
        match halt_condition {
            HaltCondition::Epochs(epochs) if epochs < 1 => {
                panic!("must train for at least one epoch")
            }
            HaltCondition::MSE(mse) if mse <= 0f64 => {
                panic!("MSE must be greater than 0")
            }
            _ => (),
        }

        self.halt_condition = halt_condition;
        self
    }

    pub fn loss_function(&mut self, loss_function: LossFunction) -> &mut Trainer<'a> {
        self.loss_function = loss_function;
        self
    }

    pub fn rate(&mut self, rate: f64) -> &mut Trainer<'a> {
        if rate <= 0f64 {
            panic!("Learning rate can't be <= 0!");
        }

        self.learning_rate = rate;
        self
    }

    pub fn momentum(&mut self, momentum: f64) -> &mut Trainer<'a> {
        if momentum < 0f64 {
            panic!("Momentum can't be < 0!");
        }

        self.momentum = momentum;
        self
    }

    pub fn log_interval(&mut self, log_interval: Option<u32>) -> &mut Trainer<'a> {
        match log_interval {
            Some(interval) if interval < 1 => {
                panic!("log interval must be Some positive number or None")
            }
            _ => (),
        }

        self.log_interval = log_interval;
        self
    }

    fn get_cross_entropy_loss(&self) -> f64 {
        let (total_error, count) =
            self.examples
                .iter()
                .fold((0.0, 0), |(acc_error, acc_count), example| {
                    let (inputs, expected_outputs) = example;
                    let (_, outputs) = self.nn.forward(inputs);
                    let output = outputs.last().unwrap();

                    let error: f64 = expected_outputs
                        .iter()
                        .zip(output)
                        .map(|(expected, actual)| {
                            let actual_clamped = actual.clamp(EPSILON, 1.0 - EPSILON);
                            -expected * actual_clamped.ln()
                                - (1.0 - expected) * (1.0 - actual_clamped).ln()
                        })
                        .sum();

                    (acc_error + error, acc_count + expected_outputs.len())
                });

        total_error / count as f64
    }

    fn get_mean_squared_error(&self) -> f64 {
        let (total_error, count) =
            self.examples
                .iter()
                .fold((0.0, 0), |(acc_error, acc_count), example| {
                    let (inputs, expected_outputs) = example;
                    let (_, outputs) = self.nn.forward(inputs);
                    let output = outputs.last().unwrap();

                    let error: f64 = expected_outputs
                        .iter()
                        .zip(output)
                        .map(|(expected, actual)| (expected - actual).powi(2))
                        .sum();

                    (acc_error + error, acc_count + expected_outputs.len())
                });

        total_error / count as f64
    }

    pub fn go(&mut self) {
        let mut current_error = 0.0;
        let mut _epochs = 0;
        let mut start = Instant::now();
        loop {
            let last_error = current_error;

            for example in self.examples {
                let (inputs, _) = example;
                let network_data = self.nn.forward(inputs);
                self.nn.backward(
                    &example,
                    network_data,
                    self.learning_rate,
                    self.momentum,
                    &mut self.prev_weights_delta,
                    &mut self.prev_biases_delta,
                );
            }

            current_error = match self.loss_function {
                LossFunction::MeanSquaredError => self.get_mean_squared_error(),
                LossFunction::CrossEntropy => self.get_cross_entropy_loss(),
            };

            let error = (last_error - current_error).abs();
            _epochs += 1;

            match self.log_interval {
                Some(interval) if _epochs % interval == 0 => {
                    println!("Epochs:{:?}; Error:{:?} ", _epochs, current_error);
                }
                _ => (),
            }
            match self.halt_condition {
                HaltCondition::Epochs(epochs_halt) => {
                    if _epochs == epochs_halt {
                        break;
                    }
                }
                HaltCondition::MSE(target_error) => {
                    if error <= target_error {
                        break;
                    }
                }
                HaltCondition::Time(time) => {
                    if start.elapsed().as_secs() >= time as u64 {
                        break;
                    }
                }
            }
        }
    }
}
