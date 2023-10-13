#![allow(clippy::type_complexity)]
mod cache;
mod dataset;
mod model;
use crate::cache::Cache;
use crate::model::GPTConfig;

use indicatif::ProgressIterator;

use dfdx::{data::*, optim::Adam, prelude::*, tensor::AutoDevice, tensor_ops::Device};
use model::GPTModel;
use rand::{
    prelude::{SeedableRng, StdRng},
    Rng,
};
use std::{fs, path::Path};

const VOCAB: usize = 256;

const SEQ_LEN: usize = 256;
const BATCH_SIZE: usize = 8;
const TEMPERATURE: f32 = 0.8;
const TOP_P: f32 = 0.95;
const TOP_K: usize = 40;
const USE_CACHE: bool = true;

type E = f32;

fn main() {
    let dev = AutoDevice::default();
    let mut rng = StdRng::seed_from_u64(0);

    let conf = GPTConfig {};

    let mut m = conf.init::<E, _>(&dev);
    let mut grads = m.alloc_grads();
    let mut opt = Adam::new(&m, Default::default());
    let mut epoch_base = 0;
    let prompt = "God";

    if let Some(path) = std::env::args().nth(1) {
        println!("load from {path}");
        let path = Path::new(&path);
        epoch_base = path
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .parse::<usize>()
            .unwrap()
            + 1;
        m.load(path).unwrap();
    }

    let train_data = dataset::DataSet::<SEQ_LEN>::new(Path::new("input.txt"));

    let preprocess = |(x, y): (Vec<u8>, Vec<u8>)| {
        let mut targets = Vec::with_capacity(VOCAB * SEQ_LEN);
        for v in y {
            let mut one_hotted: [E; VOCAB] = [0.0; VOCAB];
            one_hotted[v as usize] = E::ONE;
            targets.extend_from_slice(&one_hotted);
        }
        (
            dev.tensor_from_vec(x.iter().map(|v| *v as usize).collect(), (Const::<SEQ_LEN>,)),
            dev.tensor_from_vec(targets, (Const::<SEQ_LEN>, Const::<VOCAB>)),
        )
    };

    let start = std::time::Instant::now();
    let gen_len = SEQ_LEN;
    println!("{:}", generate(&dev, &m, prompt, gen_len, &mut rng),);
    print_metrics(start.elapsed(), gen_len);

    for i_epoch in 0..101 {
        let mut total_epoch_loss: E = 0.0;
        for (x, y) in train_data
            .shuffled(&mut rng)
            .map(preprocess)
            .batch_exact(BATCH_SIZE)
            .collate()
            .stack()
            .progress()
        {
            let y2 = m.forward_mut(x.trace(grads));
            let loss = cross_entropy_with_logits_loss(y2, y.clone());
            total_epoch_loss += loss.array();

            grads = loss.backward();
            opt.update(&mut m, &grads).unwrap();
            m.zero_grads(&mut grads);
        }

        let total_epoch = i_epoch + epoch_base;

        println!(
            "Epoch {total_epoch}, loss {:.5}, gen: {:}",
            total_epoch_loss,
            generate(&dev, &m, prompt, SEQ_LEN, &mut rng),
        );

        if i_epoch % 10 == 0 {
            fs::create_dir("save").unwrap();
            m.save(format!("save/{total_epoch}.npz"))
                .expect("fail to save model");
        }
    }
}

fn generate<E, D: Device<E>>(
    dev: &D,
    m: &GPTModel<E, D>,
    prompt: &str,
    gen_num: usize,
    rng: &mut StdRng,
) -> String
where
    E: Dtype + num_traits::Float + num_traits::AsPrimitive<f32>,
{
    let mut seq: String = prompt.into();

    let mut pos = 0;
    let seq_len = seq.len();
    let mut ids: Vec<_> = seq.as_bytes().iter().map(|v| *v as usize).collect();
    let x = dev.tensor_from_vec(ids.clone(), (seq_len,));

    let cache = if USE_CACHE {
        Some(Default::default())
    } else {
        None
    };

    let mut x_len = seq_len;
    let (mut y, mut cache) = m.try_forward(x, pos, cache).unwrap();
    pos += if cache.is_some() { x_len } else { 0 };

    for _ in 0..gen_num {
        if seq.len() >= SEQ_LEN {
            break;
        }
        //NOTE: select should be use, but the cuda select kernel will panic on my gpu, so use gather as workaround
        let probs = y.gather(dev.tensor([x_len - 1]));
        let next_idx = if TOP_K == 0 {
            greedy(probs.as_vec())
        } else {
            let probs = (probs / TEMPERATURE).softmax::<Axis<1>>().to_dtype();
            topk(probs.as_vec(), TOP_P, TOP_K, rng)
        };
        ids.push(next_idx);
        seq.push(next_idx as u8 as char);

        //next round
        let (x, pos_inc) = if cache.is_some() {
            (dev.tensor_from_vec(vec![next_idx], (1,)), 1)
        } else {
            (dev.tensor_from_vec(ids.clone(), (ids.len(),)), 0)
        };
        x_len = x.shape().0;
        (y, cache) = m.try_forward(x, pos, cache).unwrap();
        pos += pos_inc;
    }

    seq.chars()
        .map(|c| if c.is_control() { '#' } else { c })
        .collect()
}

fn greedy<E: PartialOrd>(probs: Vec<E>) -> usize {
    probs
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
        .map(|x| x.0)
        .unwrap()
}

fn topk(probs: Vec<f32>, top_p: f32, top_k: usize, rng: &mut StdRng) -> usize {
    let mut probs: Vec<_> = probs.into_iter().enumerate().collect();

    probs.sort_unstable_by(|(_, a), (_, b)| b.total_cmp(a));

    let mut choices = top_k;
    let mut total = 0.0;
    for (i, &(_, p)) in probs.iter().enumerate().take(top_k) {
        total += p;
        if total >= top_p {
            choices = i + 1;
            break;
        }
    }

    let prob: f32 = rng.gen_range(0.0..total);
    let mut accum = 0.0;
    for &(i, p) in probs.iter().take(choices) {
        accum += p;
        if accum >= prob {
            return i;
        }
    }

    unreachable!()
}

fn print_metrics(elapsed: std::time::Duration, num_tokens_generated: usize) {
    let elapsed_s = elapsed.as_secs_f64();
    let tokens_per_s = num_tokens_generated as f64 / elapsed_s;
    let ms_per_token = 1000.0 * elapsed_s / num_tokens_generated as f64;

    println!();
    println!(
        "*Generated {} tokens in {:.3?} ({tokens_per_s:.3} tokens/s, {ms_per_token:.0} ms/token)*",
        num_tokens_generated, elapsed
    );
}
