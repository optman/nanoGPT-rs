#![allow(clippy::type_complexity)]
mod cache;
mod cli;
mod dataset;
mod generate;
mod model;
mod param;
mod train;
use crate::model::GPTConfig;
use crate::{cache::Cache, generate::GenerateOption};
use cli::{Cli, Commands};
use generate::{generate, print_metrics};
use param::*;
use train::train;

use clap::Parser;
use dfdx::{prelude::*, tensor::AutoDevice};
use rand::prelude::{SeedableRng, StdRng};
use std::path::Path;

fn main() {
    let args = Cli::parse();

    let dev = AutoDevice::default();
    let mut rng = StdRng::seed_from_u64(args.seed);

    let conf = GPTConfig {};

    let mut m = conf.init::<E, _>(&dev);

    let mut epoch_base = 0;
    if let Some(path) = args.model {
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
        } => {
            let gen_opt = GenerateOption {
                use_cache: !disable_cache,
                greedy,
                top_k,
                top_p,
                temperature,
                max_seq_len: SEQ_LEN,
            };

            let start = std::time::Instant::now();
            println!(
                "{:}",
                generate(&mut rng, &dev, &m, &prompt, num_tokens, &gen_opt),
            );
            if bench {
                print_metrics(start.elapsed(), num_tokens);
            }
        }
        Commands::Train {} => {
            train(&mut rng, &dev, &mut m, epoch_base, BATCH_SIZE);
        }
    }
}
