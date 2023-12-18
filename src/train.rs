use crate::{
    dataset,
    model::{GPTModel, Params},
};
use dfdx::{data::*, nn::Adam, prelude::*};
use indicatif::{ProgressIterator, ProgressStyle};
use num_traits::ToPrimitive;
use rand::prelude::StdRng;
use std::{fs, path::Path};

#[allow(clippy::too_many_arguments)]
pub fn train<P: Params, E: Dtype, D: Device<E>>(
    rng: &mut StdRng,
    dev: &D,
    m: &mut GPTModel<P, E, D>,
    epoch_base: usize,
    batch_size: usize,
    seq_len: usize,
    input: &str,
    save_dir: &str,
    epoch_save: usize,
    epoch_max: usize,
    lr: f64,
    gen: impl Fn(&GPTModel<P, E, D>) -> String,
) where
    E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform + std::fmt::Display,
    D: TensorToArray<(), E, Array = E> + Device<f64>,
    f64: From<E>,
{
    let mut grads = m.alloc_grads();
    let mut opt: Adam<GPTModel<P, E, D>, E, D> = Adam::new(
        m,
        AdamConfig {
            lr,
            ..Default::default()
        },
    );
    let train_data = dataset::DataSet::new(Path::new(input), seq_len);

    let vocab = m.params().vocab().size();

    let preprocess = |(x, y): (&[usize], &[usize])| {
        let mut targets = vec![E::zero(); vocab * seq_len];
        for (i, v) in y.iter().enumerate() {
            targets[i * vocab + v] = E::ONE;
        }
        let x = dev.tensor_from_vec(x.to_vec(), (seq_len,));
        let y: Tensor<(usize, P::Vocab), _, _> =
            dev.tensor_from_vec(targets, (seq_len, vocab)).realize();
        (x, y)
    };

    for epoch_i in 0..epoch_max {
        let mut total_epoch_loss: E = E::zero();
        let mut total_batch = 0;
        let start = std::time::Instant::now();
        for (x, y) in train_data
            .shuffled(rng)
            .map(preprocess)
            .batch_exact(batch_size)
            .collate()
            .stack()
            .progress_with_style(
                ProgressStyle::with_template("[{elapsed_precise}] {wide_bar}  [-{eta}/{duration}]")
                    .unwrap(),
            )
        {
            let y2 = m.try_forward_mut(x.trace(grads)).unwrap();
            let loss = cross_entropy_with_logits_loss(y2, y.clone());
            total_epoch_loss += loss.array();
            total_batch += 1;

            grads = loss.backward();
            opt.update(m, &grads).unwrap();
            m.zero_grads(&mut grads);
        }

        let epoch_total = epoch_i + epoch_base;

        println!(
            "Epoch {epoch_total}, average loss {:.5}, elapsed: {:.0?} => {:}",
            total_epoch_loss.to_f64().unwrap()
                / (total_batch * batch_size * seq_len).to_f64().unwrap(),
            start.elapsed(),
            gen(m)
        );

        if epoch_i % epoch_save == 0 {
            fs::create_dir_all(save_dir).unwrap();
            m.save_safetensors(format!("{save_dir}/{epoch_total}.safetensors"))
                .expect("fail to save model");
        }
    }
}
