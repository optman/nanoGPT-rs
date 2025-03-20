use crate::{
    dataset,
    model::{GPTModel, Params},
};
use dfdx::{data::*, nn::Adam, prelude::*};
use indicatif::{ProgressIterator, ProgressStyle};
use rand::prelude::StdRng;
use std::rc::Rc;
use std::sync::RwLock;
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
    let train_data = dataset::DataSet::new(Path::new(input));

    let vocab = m.params().vocab().size();

    let answer_count_batch = Rc::new(RwLock::new(0));
    let answer_count_batch2 = answer_count_batch.clone();

    let preprocess = |line: &[usize]| {
        //search for the position of the first '=' of the line
        let pos = line.iter().position(|&c| c == b'=' as usize).unwrap();
        let answer_pos = pos + 1;

        //append '\n', and padding the line with spaces to the end with length seq_len + 1
        let mut line = line.to_owned();
        line.push(b'\n' as usize);

        let padding_start = line.len();

        let mut padding_len = seq_len - padding_start + 1;
        while padding_len > 0 {
            line.push(b' ' as usize);
            padding_len -= 1;
        }

        let answer = &line[answer_pos..padding_start];
        *answer_count_batch2.write().unwrap() += answer.len();

        //target is embeding of line[1..seq_len+ 1], but mask other regions onther than answer part.
        let mut targets = vec![E::zero(); vocab * seq_len];
        for (i, v) in answer.iter().enumerate() {
            targets[(answer_pos - 1 + i) * vocab + v] = E::ONE;
        }

        let x = dev.tensor_from_vec(line[..seq_len].to_owned(), (seq_len,));
        let y: Tensor<(usize, P::Vocab), _, _> =
            dev.tensor_from_vec(targets, (seq_len, vocab)).realize();

        (x, y)
    };

    for epoch_i in 1..=epoch_max {
        let mut answer_count_epoch = 0;
        let mut total_epoch_loss = 0.0;
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
            let loss = cross_entropy_with_logits_loss(y2, y);
            total_epoch_loss += loss.array().to_f64().unwrap();

            answer_count_epoch += *answer_count_batch2.read().unwrap();
            *answer_count_batch2.write().unwrap() = 0;

            grads = loss.backward();
            opt.update(m, &grads).unwrap();
            m.zero_grads(&mut grads);
        }

        let epoch_total = epoch_i + epoch_base;

        println!(
            "Epoch {epoch_total}, average loss {:.5}, elapsed: {:.0?} => {:}",
            total_epoch_loss / (answer_count_epoch as f64),
            start.elapsed(),
            gen(m)
        );

        if epoch_i % epoch_save == 0 || epoch_i == epoch_max {
            fs::create_dir_all(save_dir).unwrap();
            let path = format!("{save_dir}/{epoch_total}.safetensors");
            m.save_safetensors(&path).expect("fail to save model");

            println!("saved to {path}");
        }
    }
}
