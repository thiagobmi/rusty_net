mod nn;
mod trainer;

pub use nn::NN;
pub use trainer::{HaltCondition, LossFunction, Trainer};