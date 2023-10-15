use crate::{dataset, model::GPTModel, param::*};
use dfdx::{data::*, optim::Adam, prelude::*};
use indicatif::ProgressIterator;
use rand::prelude::StdRng;
use std::{fs, path::Path};

pub fn train(
    rng: &mut StdRng,
    dev: &D,
    m: &mut GPTModel<E, D>,
    epoch_base: usize,
    batch_size: usize,
) {
    let mut grads = m.alloc_grads();
    let mut opt = Adam::new(m, Default::default());
    let train_data = dataset::DataSet::<SEQ_LEN>::new(Path::new("input.txt"));

    let preprocess = |(x, y): (Vec<u8>, Vec<u8>)| {
        let mut targets = Vec::with_capacity(VOCAB * SEQ_LEN);
        for v in y {
            let mut one_hotted: [E; VOCAB] = [0.0; VOCAB];
            one_hotted[v as usize] = 1.0;
            targets.extend_from_slice(&one_hotted);
        }
        (
            dev.tensor_from_vec(x.iter().map(|v| *v as usize).collect(), (Const::<SEQ_LEN>,)),
            dev.tensor_from_vec(targets, (Const::<SEQ_LEN>, Const::<VOCAB>)),
        )
    };

    for epoch_i in 0..101 {
        let mut total_epoch_loss: E = 0.0;
        for (x, y) in train_data
            .shuffled(rng)
            .map(preprocess)
            .batch_exact(batch_size)
            .collate()
            .stack()
            .progress()
        {
            let y2 = m.forward_mut(x.trace(grads));
            let loss = cross_entropy_with_logits_loss(y2, y.clone());
            total_epoch_loss += loss.array();

            grads = loss.backward();
            opt.update(m, &grads).unwrap();
            m.zero_grads(&mut grads);
        }

        let total_epoch = epoch_i + epoch_base;

        println!("Epoch {total_epoch}, loss {:.5}", total_epoch_loss);

        if epoch_i % 10 == 0 {
            fs::create_dir_all("save").unwrap();
            m.save(format!("save/{total_epoch}.npz"))
                .expect("fail to save model");
        }
    }
}
