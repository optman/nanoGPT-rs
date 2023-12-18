#![allow(clippy::type_complexity)]
mod cache;
mod cli;
mod config;
mod dataset;
mod generate;
mod model;
mod pretokenize;
mod rotary;
mod train;
use crate::{cache::Cache, generate::GenerateOption, pretokenize::pretokenize};
use cli::{Cli, Commands};
use config::Config;
use generate::{generate, print_metrics};
use model::{GPTModel, Params};
use rust_tokenizers::tokenizer::SentencePieceBpeTokenizer;
use train::train;

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
    let tokenizer = load_tokenizer(&args.tokenizer);

    let conf = Config::load("model.json");

    match args.command {
        Commands::Generate {
            prompt,
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
                verbose: true,
                cache_size,
            };

            let start = std::time::Instant::now();
            let _ = generate(
                &tokenizer, &mut rng, &dev, &m, &prompt, num_tokens, &gen_opt,
            );
            if bench {
                print_metrics(start.elapsed(), num_tokens);
            }
        }
        Commands::Train {
            input,
            model,
            prompt,
            batch_size,
            seq_len,
            save_dir,
            epoch_save,
            epoch_max,
            lr,
        } => {
            let mut m = conf.build(&dev);
            let epoch_base = load_model(&mut m, model);

            let gen = |m: &GPTModel<_, _, _>| -> String {
                let mut rng = StdRng::seed_from_u64(0);
                generate(
                    &tokenizer,
                    &mut rng,
                    &dev,
                    m,
                    &prompt,
                    seq_len,
                    &Default::default(),
                )
            };

            train(
                &mut rng, &dev, &mut m, epoch_base, batch_size, seq_len, &input, &save_dir,
                epoch_save, epoch_max, lr, gen,
            );
        }
        Commands::PreTokenize {
            input,
            output,
            chunk_size,
        } => {
            pretokenize(&tokenizer, &input, &output, chunk_size);
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
            .unwrap()
            + 1;
        m.load_safetensors(path).unwrap();
    };

    epoch_base
}

fn load_tokenizer(path: &str) -> SentencePieceBpeTokenizer {
    SentencePieceBpeTokenizer::from_file(path, false).unwrap()
}
