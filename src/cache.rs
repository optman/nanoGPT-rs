use dfdx::{prelude::*, tensor::Tensor, tensor_ops::Device};

pub struct Cache<
    const HEADERS: usize,
    const HEADER_DIM: usize,
    const LAYERS: usize,
    E: Dtype,
    D: Device<E>,
> {
    //NOTE:my older gpu can't create zero size tensor, so use option to work around.
    pub k: [Option<Tensor<(Const<HEADERS>, usize, Const<HEADER_DIM>), E, D>>; LAYERS],
    pub v: [Option<Tensor<(Const<HEADERS>, usize, Const<HEADER_DIM>), E, D>>; LAYERS],
}

impl<
        const HEADERS: usize,
        const HEADER_DIM: usize,
        const LAYERS: usize,
        E: Dtype,
        D: Device<E>,
    > Default for Cache<HEADERS, HEADER_DIM, LAYERS, E, D>
{
    fn default() -> Self {
        Self {
            k: (0..LAYERS)
                .map(|_| None)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            v: (0..LAYERS)
                .map(|_| None)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }
}
