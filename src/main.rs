#![allow(clippy::type_complexity)]
mod cache;
mod cli;
mod dataset;
mod generate;
mod model;
mod param;
mod tokenize;
mod train;
use crate::{
    cache::Cache,
    generate::GenerateOption,
    model::GPTConfig,
    param::{D, E},
    tokenize::tokenize,
};
use cli::{Cli, Commands};
use generate::{generate, print_metrics};
use model::GPTModel;
use rust_tokenizers::tokenizer::SentencePieceBpeTokenizer;
use train::train;

use clap::Parser;
use dfdx::{prelude::*, tensor::AutoDevice};
use rand::prelude::{SeedableRng, StdRng};
use std::path::Path;

fn main() {
    let args = Cli::parse();

    let dev = AutoDevice::default();
    let mut rng = StdRng::seed_from_u64(args.seed);
    let tokenizer = load_tokenizer(&args.tokenizer);

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
        } => {
            let (m, _) = load_model(&dev, model);

            let gen_opt = GenerateOption {
                use_cache: !disable_cache,
                greedy,
                top_k,
                top_p,
                temperature,
                max_seq_len: num_tokens,
            };

            let start = std::time::Instant::now();
            println!(
                "{:}",
                generate(&tokenizer, &mut rng, &dev, &m, &prompt, num_tokens, &gen_opt)
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
        } => {
            let (mut m, epoch_base) = load_model(&dev, model);

            let gen = |m: &GPTModel<E, D>| -> String {
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
                &mut rng, &dev, &mut m, epoch_base, batch_size, seq_len, &input, gen,
            );
        }
        Commands::Tokenize {
            input,
            output,
            chunk_size,
        } => {
            tokenize(&tokenizer, &input, &output, chunk_size);
        }
    }
}

fn load_model(dev: &D, path: Option<String>) -> (GPTModel<E, D>, usize) {
    let conf = GPTConfig {};

    let mut m = conf.init::<E, _>(dev);

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
        m.load(path).unwrap();
    };

    (m, epoch_base)
}

fn load_tokenizer(path: &str) -> SentencePieceBpeTokenizer {
    SentencePieceBpeTokenizer::from_file(path, false).unwrap()
}
