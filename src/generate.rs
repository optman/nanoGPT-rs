use crate::{
    cache::Cache,
    model::{GPTModel, Params},
};
use dfdx::prelude::*;
use rand::{rngs::StdRng, Rng};
use rust_tokenizers::{
    tokenizer::{Tokenizer, TruncationStrategy},
    vocab::Vocab,
};
use std::io::Write;

pub struct GenerateOption {
    pub greedy: bool,
    pub use_cache: bool,
    pub top_k: usize,
    pub top_p: f32,
    pub temperature: f32,
    pub max_seq_len: usize,
    pub pos_scale: usize,
    pub verbose: bool,
    pub cache_size: usize,
}

impl Default for GenerateOption {
    fn default() -> Self {
        Self {
            greedy: false,
            use_cache: true,
            top_k: 40,
            top_p: 0.95,
            temperature: 0.8,
            max_seq_len: 100,
            pos_scale: 1,
            verbose: false,
            cache_size: 256,
        }
    }
}

pub fn generate<P: Params, V: Vocab, T: Tokenizer<V>, E, D: Device<E>>(
    tokenizer: &T,
    rng: &mut StdRng,
    dev: &D,
    m: &GPTModel<P, E, D>,
    prompt: &str,
    gen_num: usize,
    opt: &GenerateOption,
) -> String
where
    E: Dtype + num_traits::Float + num_traits::AsPrimitive<f32>,
    f64: From<E>,
    D: Device<f64>,
{
    if opt.verbose {
        print!("{:}", prompt);
        std::io::stdout().flush().unwrap();
    }

    let prompt = tokenizer
        .encode(
            prompt,
            None,
            prompt.len(),
            &TruncationStrategy::DoNotTruncate,
            0,
        )
        .token_ids;
    let mut seq: Vec<usize> = prompt.into_iter().map(|c| c as usize).collect();

    let mut pos = 0;
    let seq_len = seq.len();
    let x = dev.tensor_from_vec(seq.clone(), (seq_len,));

    let cache = if opt.use_cache {
        Some(Cache::new(m.params().layers(), opt.cache_size))
    } else {
        None
    };

    let mut x_len = seq_len;
    let (mut y, mut cache) = m.try_forward(x, pos, opt.pos_scale, cache).unwrap();
    pos += if cache.is_some() { x_len } else { 0 };

    for _ in 0..gen_num {
        if seq.len() >= opt.max_seq_len {
            break;
        }
        //NOTE: select should be use, but the cuda select kernel will panic on my gpu, so use gather as workaround
        let probs = y.gather(dev.tensor([x_len - 1]));
        let next_idx = if opt.greedy {
            greedy(probs.as_vec())
        } else {
            let probs = (probs / opt.temperature).softmax::<Axis<1>>().to_dtype();
            topk(probs.as_vec(), opt.top_p, opt.top_k, rng)
        };
        seq.push(next_idx);

        if opt.verbose {
            print!("{:}", tokenizer.decode(&[next_idx as i64], true, false));
            std::io::stdout().flush().unwrap();
        }

        //next round
        let (x, pos_inc) = if cache.is_some() {
            (dev.tensor_from_vec(vec![next_idx], (1,)), 1)
        } else {
            (dev.tensor_from_vec(seq.clone(), (seq.len(),)), 0)
        };
        x_len = x.shape().0;
        (y, cache) = m.try_forward(x, pos, opt.pos_scale, cache).unwrap();
        pos += pos_inc;
    }

    tokenizer.decode(
        &seq.into_iter().map(|c| c as i64).collect::<Vec<_>>(),
        true,
        false,
    )
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

pub fn print_metrics(elapsed: std::time::Duration, num_tokens_generated: usize) {
    let elapsed_s = elapsed.as_secs_f64();
    let tokens_per_s = num_tokens_generated as f64 / elapsed_s;
    let ms_per_token = 1000.0 * elapsed_s / num_tokens_generated as f64;

    println!();
    println!(
        "*Generated {} tokens in {:.3?} ({tokens_per_s:.3} tokens/s, {ms_per_token:.0} ms/token)*",
        num_tokens_generated, elapsed
    );
}
