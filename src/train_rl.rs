use crate::{
    dataset,
    generate::{generate, GenerateOption},
    model::{GPTModel, Params},
};
use dfdx::{data::*, nn::Adam, prelude::*};
use dfdx_core::tensor_ops;
use indicatif::{ProgressIterator, ProgressStyle};
use num_traits::ToPrimitive;
use rand::{prelude::StdRng, SeedableRng};
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
    rollout_num: usize,
    rollout_temperature: f32,
    rollout_dump: usize,
    clip_ratio: f32,
    pi_iters: usize,
    kl_target: f32,
    gen: impl Fn(&GPTModel<P, E, D>) -> String,
) where
    E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform + std::fmt::Display,
    E: num_traits::AsPrimitive<f32>,
    D: TensorToArray<(), E, Array = E> + Device<f64>,
    D: tensor_ops::ReshapeKernel<usize>,
    D: tensor_ops::SliceKernel<usize>,
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

    let m = Rc::new(RwLock::new(m));

    let rollout_model = m.clone();

    let total_batch_reward = Rc::new(RwLock::new(0.0));
    let total_batch_reward2 = total_batch_reward.clone();

    let preprocess_count = Rc::new(RwLock::new(0));

    let preprocess = |line: &[usize]| {
        *preprocess_count.write().unwrap() += 1;

        //search for the position of the first '=' of the line
        let pos = line.iter().position(|&c| c == b'=' as usize).unwrap();
        let question = &line[..pos + 1];
        //search for the position of the last '=' of the line
        let pos = line.iter().rposition(|&c| c == b'=' as usize).unwrap();
        let answer = &line[pos + 1..];
        //convert answer back to string, and then to usize
        let answer = String::from_utf8(answer.iter().map(|&c| c as u8).collect()).unwrap();

        //convert question into string
        let prompt = String::from_utf8(question.iter().map(|&c| c as u8).collect()).unwrap();
        //roll out
        let gen_opt = GenerateOption {
            max_seq_len: seq_len,
            temperature: rollout_temperature,
            log_probs: true,
            ..Default::default()
        };

        let mut rollouts = Vec::new();
        let mut rewards = Vec::new();
        let mut masks = Vec::new();
        let mut logps = Vec::new();

        let dump_rollout =
            rollout_dump > 0 && *preprocess_count.read().unwrap() % rollout_dump == 0;

        let (completion, logp) = generate(
            &mut StdRng::from_entropy(),
            dev,
            &rollout_model.read().unwrap(),
            vec![&prompt; rollout_num],
            seq_len,
            &gen_opt,
        );
        let logp = logp.unwrap();

        for (completion, _logp) in completion.into_iter().zip(logp.into_iter()) {
            let mut mask = vec![0.0; prompt.len()];
            let mut logp = vec![1.0; prompt.len()];

            let rollout = format!("{prompt}{completion}");
            if dump_rollout {
                println!("{:?}", rollout);
            }

            for _ in 0..completion.len() {
                mask.push(1.0);
            }
            logp.extend(_logp);

            let reward = compute_reward(&completion, &answer);

            let mut rollout: Vec<usize> = rollout.chars().map(|c| c as usize).collect();
            while rollout.len() < seq_len {
                rollout.push(b' ' as usize);
                mask.push(0.0);
                logp.push(1.0);
            }

            rollouts.push(rollout);
            rewards.push(reward);
            masks.push(mask);
            logps.push(logp);
        }

        *total_batch_reward2.write().unwrap() += rewards.iter().sum::<f64>();

        let advantages = normalize_rewards(&rewards);
        let rollouts = rollouts.into_iter().flat_map(|r| r.into_iter()).collect();
        let masks = masks
            .into_iter()
            .flat_map(|m| m.into_iter())
            .map(|n| E::from_f64(n).unwrap())
            .collect();
        let logps = logps
            .into_iter()
            .flat_map(|r| r.into_iter())
            .map(|n| E::from_f64(n).unwrap())
            .collect();

        let rollouts = dev.tensor_from_vec(rollouts, (rollout_num, seq_len));
        let advantages = dev.tensor_from_vec(advantages, (rollout_num,));
        let masks = dev.tensor_from_vec(masks, (rollout_num, seq_len));
        let log_probs = dev.tensor_from_vec(logps, (rollout_num, seq_len));
        ((rollouts, advantages), (masks, log_probs))
    };

    for epoch_i in 1..=epoch_max {
        let mut total_epoch_loss: E = E::zero();
        let mut total_batch = 0;
        let mut total_epoch_reward = 0.0;
        let start = std::time::Instant::now();
        for (rollouts_advantages, masks_probs) in train_data
            .shuffled(rng)
            .map(preprocess)
            .batch_exact(batch_size)
            .collate()
            .progress_with_style(
                ProgressStyle::with_template("[{elapsed_precise}] {wide_bar}  [-{eta}/{duration}]")
                    .unwrap(),
            )
        {
            let (rollouts, advantages): (Vec<_>, Vec<_>) = rollouts_advantages.into_iter().unzip();
            let rollouts = rollouts.stack();
            let advantages = advantages.stack();
            let (masks, probs): (Vec<_>, Vec<_>) = masks_probs.into_iter().unzip();
            let masks = masks.stack();
            let probs = probs.stack();

            total_batch += 1;
            total_epoch_reward += *total_batch_reward.read().unwrap();
            *total_batch_reward.write().unwrap() = 0.0;

            let rollouts = rollouts
                .try_reshape_like(&(batch_size * rollout_num, seq_len))
                .unwrap();
            let advantages = advantages
                .try_reshape_like(&(batch_size * rollout_num,))
                .unwrap();
            let masks = masks
                .try_reshape_like(&(batch_size * rollout_num, seq_len))
                .unwrap();
            let probs = probs
                .try_reshape_like(&(batch_size * rollout_num, seq_len))
                .unwrap();

            for i in 0..pi_iters {
                let m = &mut m.write().unwrap();
                let (loss, kl) = compute_loss(
                    grads,
                    m,
                    rollouts.clone(),
                    advantages.clone(),
                    masks.clone(),
                    probs.clone(),
                    clip_ratio,
                );

                if kl.abs() as f32 > kl_target {
                    println!("early stop at step {i} due to reaching max kl {:.5}", kl);
                    grads = m.alloc_grads();
                    break;
                }

                total_epoch_loss += loss.array();

                grads = loss.backward();
                opt.update(m, &grads).unwrap();
                m.zero_grads(&mut grads);
            }
        }

        let epoch_total = epoch_i + epoch_base;

        println!(
            "Epoch {epoch_total}, average reward {:.5}, average loss {:.5}, elapsed: {:.0?} => {:}",
            total_epoch_reward / (total_batch * batch_size * rollout_num).to_f64().unwrap(),
            total_epoch_loss.to_f64().unwrap() / (total_batch * pi_iters).to_f64().unwrap(),
            start.elapsed(),
            gen(&m.read().unwrap())
        );

        if epoch_i % epoch_save == 0 || epoch_i == epoch_max {
            fs::create_dir_all(save_dir).unwrap();
            let path = format!("{save_dir}/{epoch_total}.safetensors");
            m.write()
                .unwrap()
                .save_safetensors(&path)
                .expect("fail to save model");

            println!("saved to {path}");
        }
    }
}

fn compute_loss<Batch: Dim, Seq: Dim, P: Params, E: Dtype, D: Device<E>>(
    grads: Gradients<E, D>,
    m: &mut GPTModel<P, E, D>,
    rollout: Tensor<(Batch, Seq), usize, D, NoneTape>,
    advantages: Tensor<(Batch,), E, D, NoneTape>,
    masks: Tensor<(Batch, Seq), E, D, NoneTape>,
    logps: Tensor<(Batch, Seq), E, D, NoneTape>,
    clip_ratio: f32,
) -> (Tensor<(), E, D, OwnedTape<E, D>>, f64)
where
    D: dfdx::prelude::Device<f64>,
    E: num_traits::Float,
    D: TensorToArray<(), E, Array = E> + Device<f64>,
    D: tensor_ops::SliceKernel<usize>,
    f64: From<E>,
{
    let seq_len = rollout.shape().1.size();
    let x = rollout.clone().slice((.., 0..seq_len - 1));
    let y = rollout.slice((.., 1..));
    let masks = masks.slice((.., 1..));
    let logps_old = logps.slice((.., 1..));

    let logits = m.try_forward_mut(x.trace(grads)).unwrap();
    let logps = logits.log_softmax::<Axis<2>>();
    let logps = logps.select(y.with_empty_tape());

    let masks_percent =
        masks.clone().sum().array().to_f64().unwrap() / masks.len().to_f64().unwrap();

    if clip_ratio > 0.0 {
        //PPO
        let approx_kl = ((logps.with_empty_tape() - logps_old.clone()) * masks.clone())
            .mean()
            .array()
            .to_f64()
            .unwrap()
            / masks_percent;

        let ratios = (logps - logps_old).exp();
        let advantages = advantages.broadcast_like(&ratios);

        let surr1 = ratios.with_empty_tape() * advantages.clone() * masks.clone();
        let surr2 = ratios.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages.clone() * masks;

        (
            surr2.minimum(surr1).mean().negate() / masks_percent,
            approx_kl,
        )
    } else {
        //VPG
        let advantages = advantages.broadcast_like(&logps);
        let losses = logps * advantages * masks;

        (losses.mean().negate() / masks_percent, 0.0)
    }
}

fn compute_reward(compleption: &str, answer: &str) -> f64 {
    let guess = compleption.rsplitn(2, '=').next().unwrap();
    if guess == answer {
        1.0
    } else {
        -1.0
    }
}

fn normalize_rewards<E: Dtype>(rewards: &Vec<f64>) -> Vec<E> {
    // Calculate mean
    let mean: f64 = rewards.iter().sum::<f64>() / rewards.len() as f64;

    // Calculate standard deviation
    let variance: f64 =
        rewards.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / rewards.len() as f64;
    let std = variance.sqrt();

    // Normalize rewards
    rewards
        .iter()
        .map(|x| (x - mean) / (std + 1e-4))
        .map(|n| E::from_f64(n).unwrap())
        .collect()
}
