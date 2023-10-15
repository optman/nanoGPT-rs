use crate::model::GPTModel;
use dfdx::{prelude::*, tensor_ops::Device};
use rand::{rngs::StdRng, Rng};

pub struct GenerateOption {
    pub greedy: bool,
    pub use_cache: bool,
    pub top_k: usize,
    pub top_p: f32,
    pub temperature: f32,
    pub max_seq_len: usize,
}

pub fn generate<E, D: Device<E>>(
    rng: &mut StdRng,
    dev: &D,
    m: &GPTModel<E, D>,
    prompt: &str,
    gen_num: usize,
    opt: &GenerateOption,
) -> String
where
    E: Dtype + num_traits::Float + num_traits::AsPrimitive<f32>,
{
    let mut seq: String = prompt.into();

    let mut pos = 0;
    let seq_len = seq.len();
    let mut ids: Vec<_> = seq.as_bytes().iter().map(|v| *v as usize).collect();
    let x = dev.tensor_from_vec(ids.clone(), (seq_len,));

    let cache = if opt.use_cache {
        Some(Default::default())
    } else {
        None
    };

    let mut x_len = seq_len;
    let (mut y, mut cache) = m.try_forward(x, pos, cache).unwrap();
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
