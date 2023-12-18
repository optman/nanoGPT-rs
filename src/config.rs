use crate::{
    model::{BlockConfig, GPTModel, GPTModelConfig, Params},
    rotary::RotaryEmbeddingConfig,
};
use dfdx::prelude::*;
use rand_distr::{uniform::SampleUniform, Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, path::Path};

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Config {
    pub vocab: usize,
    pub hidden: usize,
    pub mlp_hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub kv_heads: usize,
    pub kv_dim: usize,
    pub layers: usize,
    pub dropout: f64,
    pub rope_base: i64,
    pub rope_max_seq: usize,
}

impl Config {
    pub fn build<E: Dtype, D: Device<E>>(self, dev: &D) -> GPTModel<Self, E, D>
    where
        E: SampleUniform + num_traits::Float,
        StandardNormal: Distribution<E>,
    {
        let m = GPTModelConfig {
            embedding_layer: EmbeddingConfig {
                vocab: self.vocab(),
                model: self.hidden(),
            },
            pos_enc: RotaryEmbeddingConfig {
                head_dim: self.head_dim(),
                max_seq: self.rope_max_seq,
                base: self.rope_base,
            },
            dropout: Dropout { p: self.dropout() },
            atten_layers: (0..self.layers().size())
                .map(|_| BlockConfig::new(self))
                .collect(),
            lm_header: LinearConfig {
                inp: self.hidden(),
                out: self.vocab(),
            },
            params: self,
        };

        dev.build_module(m)
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Self {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }
}

impl Params for Config {
    type Vocab = usize;
    type Hidden = usize;
    type MlpHidden = usize;
    type Heads = usize;
    type HeadDim = usize;
    type KvHeads = usize;
    type KvDim = usize;
    type Layers = usize;

    fn vocab(&self) -> Self::Vocab {
        self.vocab
    }
    fn hidden(&self) -> Self::Hidden {
        self.hidden
    }

    fn mlp_hidden(&self) -> Self::MlpHidden {
        self.mlp_hidden
    }

    fn heads(&self) -> Self::Heads {
        self.heads
    }

    fn head_dim(&self) -> Self::HeadDim {
        self.head_dim
    }

    fn kv_heads(&self) -> Self::KvHeads {
        self.kv_heads
    }

    fn kv_dim(&self) -> Self::KvDim {
        self.kv_dim
    }

    fn layers(&self) -> Self::Layers {
        self.layers
    }

    fn dropout(&self) -> f64 {
        self.dropout
    }
}
