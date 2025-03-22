#![allow(clippy::type_complexity)]
mod cache;
mod cli;
mod config;
mod dataset;
mod eval;
mod generate;
mod model;
mod rotary;
mod train;
mod train_rl;
mod train_sft;
use crate::{cache::Cache, generate::GenerateOption};
use cli::{Cli, Commands, TrainMethod};
use config::Config;
use eval::eval;
use generate::{generate, print_metrics};
use model::{GPTModel, Params};
use train::train as pre_train;
use train_rl::train as train_rl;
use train_sft::train as sft_train;

use clap::Parser;
use dfdx::{nn::LoadSafeTensors, tensor::AutoDevice};
use rand::prelude::{SeedableRng, StdRng};
use std::path::Path;

type E = f32;
type D = AutoDevice;

fn main() {
    let args = Cli::parse();

    let dev = AutoDevice::default();
    let mut rng = StdRng::seed_from_u64(args.seed);

    let conf = Config::load("model.json");

    match args.command {
        Commands::Eval {
            input,
            model,
            batch_size,
            num_tokens,
        } => {
            let mut m = conf.build(&dev);
            load_model(&mut m, model);

            let (total, acc) = eval(&dev, &m, &input, batch_size, num_tokens);

            println!(
                "total: {total}, correct: {acc}, accuracy: {:.0}%",
                acc as f32 / total as f32 * 100.0
            );
        }
        Commands::Generate {
            prompts,
            disable_cache,
            top_k,
            top_p,
            temperature,
            num_tokens,
            greedy,
            bench,
            model,
            pos_scale,
            cache_size,
        } => {
            let mut m = conf.build(&dev);
            load_model(&mut m, model);

            let gen_opt = GenerateOption {
                use_cache: !disable_cache,
                greedy,
                top_k,
                top_p,
                temperature,
                max_seq_len: num_tokens,
                pos_scale,
                verbose: false,
                cache_size,
                ..Default::default()
            };

            let start = std::time::Instant::now();
            let (completes, _) = generate(
                &mut rng,
                &dev,
                &m,
                prompts.iter().map(|x| x.as_str()).collect(),
                num_tokens,
                &gen_opt,
            );

            prompts.iter().zip(completes.iter()).for_each(|(p, c)| {
                println!("{p}{c}");
            });

            if bench {
                print_metrics(start.elapsed(), num_tokens);
            }
        }
        Commands::Train {
            input,
            model,
            prompts,
            batch_size,
            seq_len,
            save_dir,
            epoch_save,
            epoch_max,
            lr,
            method,
        } => {
            let mut m = conf.build(&dev);
            let epoch_base = load_model(&mut m, model);

            let gen = |m: &GPTModel<_, _, _>| -> String {
                let mut rng = StdRng::seed_from_u64(0);
                let gen_opt = GenerateOption {
                    max_seq_len: seq_len,
                    greedy: true,
                    ..Default::default()
                };
                let (completions, _) = generate(
                    &mut rng,
                    &dev,
                    m,
                    prompts.iter().map(|x| x.as_str()).collect(),
                    seq_len,
                    &gen_opt,
                );

                prompts
                    .iter()
                    .zip(completions.iter())
                    .fold(String::new(), |acc, (p, c)| format!("{acc} {p}{c}"))
            };

            match method {
                TrainMethod::Pretrain {} => {
                    pre_train(
                        &mut rng, &dev, &mut m, epoch_base, batch_size, seq_len, &input, &save_dir,
                        epoch_save, epoch_max, lr, gen,
                    );
                }
                TrainMethod::SFT {} => {
                    sft_train(
                        &mut rng, &dev, &mut m, epoch_base, batch_size, seq_len, &input, &save_dir,
                        epoch_save, epoch_max, lr, gen,
                    );
                }
                TrainMethod::RL {
                    rollout_num,
                    rollout_temperature,
                    rollout_dump,
                    clip_ratio,
                    pi_iters,
                    kl_target,
                } => {
                    train_rl(
                        &mut rng,
                        &dev,
                        &mut m,
                        epoch_base,
                        batch_size,
                        seq_len,
                        &input,
                        &save_dir,
                        epoch_save,
                        epoch_max,
                        lr,
                        rollout_num,
                        rollout_temperature,
                        rollout_dump,
                        clip_ratio,
                        pi_iters,
                        kl_target,
                        gen,
                    );
                }
            }
        }
    }
}

fn load_model<P: Params>(m: &mut GPTModel<P, E, D>, path: Option<String>) -> usize {
    let mut epoch_base = 0;
    if let Some(path) = path {
        println!("load from {path}");
        let path = Path::new(&path);
        epoch_base = path
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .parse::<usize>()
            .unwrap();
        m.load_safetensors(path).unwrap();
    };

    epoch_base
}
