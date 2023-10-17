use itertools::Itertools;
use rust_tokenizers::{
    tokenizer::{Tokenizer, TruncationStrategy},
    vocab::Vocab,
};
use std::io::{Read, Write};

pub fn tokenize<V: Vocab, T: Tokenizer<V>>(
    tokenizer: &T,
    input: &str,
    output: &str,
    chunk_size: usize,
) {
    let mut buf: String = Default::default();
    std::fs::File::open(input)
        .unwrap()
        .read_to_string(&mut buf)
        .unwrap();
    let mut output = std::fs::File::create(output).unwrap();

    for chunk in &buf.chars().chunks(chunk_size) {
        let chunk: String = chunk.collect();
        let tokens = tokenizer.encode(
            &chunk,
            None,
            chunk.len(),
            &TruncationStrategy::DoNotTruncate,
            0,
        );
        let buf: Vec<u8> = tokens
            .token_ids
            .into_iter()
            .flat_map(|c| (c as u16).to_be_bytes())
            .collect();
        let _ = output.write(&buf).unwrap();
    }
}
