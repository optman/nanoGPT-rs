use crate::{
    generate::generate,
    model::{GPTModel, Params},
};
use dfdx::prelude::*;
use itertools::Itertools;
use rand::{rngs::StdRng, SeedableRng};
use std::io::BufRead;

pub(crate) fn eval<P: Params, E: Dtype, D: Device<E>>(
    dev: &D,
    m: &GPTModel<P, E, D>,
    input_file: &str,
    batch_size: usize,
    num_tokens: usize,
) -> (usize, usize)
where
    E: num_traits::Float + num_traits::AsPrimitive<f32>,
    D: Device<f64>,
    f64: From<E>,
{
    let f = std::fs::File::open(input_file).unwrap();
    let reader = std::io::BufReader::new(f);

    reader
        .lines()
        .map(|line| {
            let line = line.unwrap();
            let pos = line.chars().position(|c| c == '=').unwrap();
            (line[..pos + 1].to_owned(), line[pos + 1..].to_owned())
        })
        .chunks(batch_size)
        .into_iter()
        .map(|batch| {
            let (questions, answers): (Vec<_>, Vec<_>) = batch.unzip();
            let (completions, _) = generate(
                &mut StdRng::from_entropy(),
                dev,
                m,
                questions.iter().map(|s| s.as_str()).collect_vec(),
                num_tokens,
                &Default::default(),
            );

            answers
                .iter()
                .zip(completions.iter())
                .map(|(a, c)| {
                    let a = a.trim();
                    let c = c.trim();
                    if a == c {
                        1
                    } else {
                        0
                    }
                })
                .fold((0, 0), |(total, acc), x| (total + 1, acc + x))
        })
        .fold((0, 0), |(total, acc), (t, a)| (total + t, acc + a))
}
