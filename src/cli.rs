use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Commands,

    #[arg(long, default_value_t = 0)]
    pub(crate) seed: u64,

    #[arg(short, long, default_value = "tokenizer.model")]
    pub(crate) tokenizer: String,
}

#[derive(Subcommand, Debug)]
pub(crate) enum Commands {
    Generate {
        #[arg(short, long, default_value = "God")]
        prompt: String,

        #[arg(short, long, default_value_t = 256)]
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
    },

    Train {
        #[arg(short, long, default_value = "input.bin")]
        input: String,

        #[arg(short, long, default_value=None)]
        model: Option<String>,

        #[arg(short, long, default_value = "God")]
        prompt: String,

        #[arg(short, long, default_value_t = 8)]
        batch_size: usize,

        #[arg(short, long, default_value_t = 64)]
        seq_len: usize,
    },
    Tokenize {
        #[arg(short, long, default_value = "")]
        input: String,

        #[arg(short, long, default_value = "")]
        output: String,

        #[arg(short, long, default_value_t = 10_000)]
        chunk_size: usize,
    },
}
