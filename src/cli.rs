use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Commands,

    #[arg(long, default_value_t = 0)]
    pub(crate) seed: u64,
}

#[derive(Subcommand, Debug)]
pub(crate) enum Commands {
    Generate {
        #[arg(short, long, default_value = "1+1=")]
        prompts: Vec<String>,

        #[arg(short, long, default_value_t = 16)]
        num_tokens: usize,

        #[arg(long, default_value_t = false)]
        disable_cache: bool,

        #[arg(long, default_value_t = false)]
        greedy: bool,

        #[arg(long, default_value_t = 0.95)]
        top_p: f32,

        #[arg(long, default_value_t = 40)]
        top_k: usize,

        #[arg(long, default_value_t = 0.8)]
        temperature: f32,

        #[arg(long, default_value_t = false)]
        bench: bool,

        #[arg(short, long, default_value=None)]
        model: Option<String>,

        #[arg(long, default_value_t = 1)]
        pos_scale: usize,

        #[arg(long, default_value_t = 16)]
        cache_size: usize,
    },

    Train {
        #[arg(short, long, default_value = "input.txt")]
        input: String,

        #[arg(short, long, default_value=None)]
        model: Option<String>,

        #[arg(short, long, default_value = "1+1=")]
        prompts: Vec<String>,

        #[arg(short, long, default_value_t = 8)]
        batch_size: usize,

        #[arg(short, long, default_value_t = 16)]
        seq_len: usize,

        #[arg(long, default_value = "save")]
        save_dir: String,

        #[arg(long, default_value_t = 10)]
        epoch_save: usize,

        #[arg(long, default_value_t = 100)]
        epoch_max: usize,

        #[arg(long, default_value_t = 1e-5)]
        lr: f64,

        #[command(subcommand)]
        method: TrainMethod,
    },
}

#[derive(Subcommand, Debug)]
pub(crate) enum TrainMethod {
    Pretrain {},
    SFT {},
    RL {
        #[arg(long, default_value_t = 8)]
        rollout_num: usize,

        #[arg(long, default_value_t = 1.0)]
        rollout_temperature: f32,

        #[arg(long, default_value_t = 0)]
        rollout_dump: usize,

        #[arg(long, default_value_t = 0.2)]
        clip_ratio: f32,

        #[arg(long, default_value_t = 1)]
        pi_iters: usize,

        #[arg(long, default_value_t = 0.01)]
        kl_target: f32,
    },
}
