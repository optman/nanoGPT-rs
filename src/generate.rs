use crate::{
    cache::Cache,
    model::{GPTModel, Params},
};
use dfdx::prelude::*;
use rand::{rngs::StdRng, Rng};
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
    pub log_probs: bool,
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
            log_probs: false,
        }
    }
}

pub fn generate<P: Params, E, D: Device<E>>(
    rng: &mut StdRng,
    dev: &D,
    m: &GPTModel<P, E, D>,
    prompts: Vec<&str>,
    gen_num: usize,
    opt: &GenerateOption,
) -> (Vec<String>, Option<Vec<Vec<f64>>>)
where
    E: Dtype + num_traits::Float + num_traits::AsPrimitive<f32>,
    f64: From<E>,
    D: Device<f64>,
{
    //only print the first prompt
    if opt.verbose {
        print!("{:}", prompts[0]);
        std::io::stdout().flush().unwrap();
    }

    /*
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
    */

    let mut log_probs = Vec::new();

    let mut seqs = Vec::new();
    let mut poss = Vec::new();
    let mut pad_starts = Vec::new();
    let mut ends = Vec::new();
    let mut stops = vec![false; prompts.len()];

    let mut seq_len = prompts.iter().map(|p| p.len()).max().unwrap();
    let pad_end = seq_len;

    for prompt in &prompts {
        let mut seq: Vec<usize> = prompt.chars().map(|c| c as usize).collect();
        let mut pos: Vec<usize> = (0..seq.len()).collect();
        pad_starts.push(seq.len());
        let last_pos = seq.len() - 1;
        while seq.len() < seq_len {
            seq.push(b' ' as usize);
            pos.push(last_pos);
        }

        ends.push(seq.len());
        seqs.push(seq);
        poss.push(pos);

        log_probs.push(Vec::new());
    }
    let batch = prompts.len();
    let x = dev.tensor_from_vec(
        seqs.clone()
            .into_iter()
            .flat_map(|s| s.into_iter())
            .collect(),
        (batch, seq_len),
    );

    let mut cache = if opt.use_cache {
        Some(Cache::new(m.params().layers(), opt.cache_size))
    } else {
        None
    };

    let mut x_len = seq_len;
    let pos = gen_pos::<E, D>(dev, &poss, x_len, opt.pos_scale);
    let mut y = m.try_forward(x, pos, &mut cache.as_mut()).unwrap();

    for i in 0..gen_num {
        if seq_len >= opt.max_seq_len {
            break;
        }
        //check all complete?
        if stops.iter().all(|&x| x) {
            break;
        }

        let new_pos = if i == 0 {
            pad_starts.iter().map(|x| x - 1).collect()
        } else {
            vec![x_len - 1; batch]
        };

        let new_pos = dev.tensor_from_vec(new_pos, (batch,));
        let batch_logits = y.select(new_pos);
        let mut next_idxs = Vec::new();
        for i in 0..batch {
            let last_pos = *poss[i].last().unwrap();
            if stops[i] {
                seqs[i].push(b' ' as usize); //paddings at the end
                next_idxs.push(b' ' as usize);
                poss[i].push(last_pos);
                continue;
            }

            let logits = batch_logits.clone().select(dev.tensor(i));
            let next_idx = if opt.greedy {
                greedy(logits.as_vec())
            } else {
                let probs = (logits.clone() / opt.temperature)
                    .softmax::<Axis<0>>()
                    .to_dtype()
                    .as_vec();
                let idx = topk(&probs, opt.top_p, opt.top_k, rng);
                idx
            };

            next_idxs.push(next_idx);

            if next_idx == '\n' as usize {
                seqs[i].push(b' ' as usize); //paddings at the end
                poss[i].push(last_pos);
                stops[i] = true;
                continue;
            }

            seqs[i].push(next_idx);
            poss[i].push(last_pos + 1);
            ends[i] += 1;

            if opt.log_probs {
                log_probs[i].push(logits.log_softmax::<Axis<0>>().to_dtype().as_vec()[next_idx]);
            }

            //only print the first completion
            if opt.verbose && i == 0 {
                //print!("{:}", tokenizer.decode(&[next_idx as i64], true, false));
                print!("{:}", std::char::from_u32(next_idx as u32).unwrap());
                std::io::stdout().flush().unwrap();
            }
        }
        seq_len += 1;

        //next round
        let x = if cache.is_some() {
            dev.tensor_from_vec(next_idxs, (batch, 1))
        } else {
            dev.tensor_from_vec(
                seqs.clone()
                    .into_iter()
                    .flat_map(|s| s.into_iter())
                    .collect(),
                (batch, seq_len),
            )
        };

        x_len = x.shape().1;
        let pos = gen_pos::<E, D>(dev, &poss, x_len, opt.pos_scale);
        y = m.try_forward(x, pos, &mut cache.as_mut()).unwrap();
    }

    /*
    tokenizer.decode(
        &seq.into_iter().map(|c| c as i64).collect::<Vec<_>>(),
        true,
        false,
    )
    */

    let mut completions = Vec::new();
    seqs.into_iter().enumerate().for_each(|(i, s)| {
        let s = String::from_utf8(s[pad_end..ends[i]].iter().map(|&c| c as u8).collect()).unwrap();
        completions.push(s);
    });

    let log_probs = if opt.log_probs { Some(log_probs) } else { None };

    (completions, log_probs)
}

fn gen_pos<E: Dtype, D: Device<E>>(
    dev: &D,
    pos: &Vec<Vec<usize>>,
    len: usize,
    pos_scale: usize,
) -> Tensor<(usize, usize), usize, D> {
    pos.iter()
        .map(|pos| {
            let pos: Vec<_> = pos
                .iter()
                .skip(pos.len() - len)
                .map(|x| x / pos_scale)
                .collect();

            dev.tensor_from_vec(pos, (len,))
        })
        .collect::<Vec<_>>()
        .stack()
}

fn greedy<E: PartialOrd>(logits: Vec<E>) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
        .map(|x| x.0)
        .unwrap()
}

fn topk(probs: &Vec<f32>, top_p: f32, top_k: usize, rng: &mut StdRng) -> usize {
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
