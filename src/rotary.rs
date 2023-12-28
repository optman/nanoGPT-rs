use dfdx::prelude::*;

#[derive(Clone, Debug, Default)]
pub struct RotaryEmbeddingConfig<HeadDim: Dim> {
    pub head_dim: HeadDim,
    pub max_seq: usize,
    pub base: i64,
}

impl<HeadDim: Dim, E: Dtype, D: Device<E>> BuildOnDevice<E, D> for RotaryEmbeddingConfig<HeadDim> {
    type Built = RotaryEmbedding<HeadDim, E, D>;

    fn try_build_on_device(&self, dev: &D) -> Result<Self::Built, dfdx_core::tensor::Error> {
        Ok(new(self.head_dim, self.max_seq, self.base, dev))
    }
}

#[derive(Clone, Debug, ZeroGrads, UpdateParams, ResetParams, SaveSafeTensors, LoadSafeTensors)]
pub struct RotaryEmbedding<HeadDim: Dim, E: Dtype, D: Device<E>> {
    head_dim: HeadDim,
    cos: Tensor<(usize, HeadDim), E, D>,
    sin: Tensor<(usize, HeadDim), E, D>,
}
impl<HeadDim: Dim, E: Dtype, D: Device<E>> RotaryEmbedding<HeadDim, E, D> {
    //none tape
    pub fn try_forward<Seq: Dim, Headers: Dim>(
        &self,
        x: Tensor<(Seq, Headers, HeadDim), E, D>,
        pos: usize,
        pos_scale: usize,
    ) -> Result<Tensor<(Seq, Headers, HeadDim), E, D>, Error> {
        let seq = x.shape().0;
        let half_hiden = self.head_dim.size() / 2;
        let first_half = x.clone().slice((.., .., 0..half_hiden));
        let second_half = x.clone().slice((.., .., half_hiden..));

        let idx = x.dev().tensor_from_vec(
            (pos..pos + seq.size()).map(|n| n / pos_scale).collect(),
            (seq,),
        );

        let sub_cos: Tensor<(Seq, HeadDim), _, _> = self.cos.clone().gather(idx.clone()).realize();
        let sub_sin: Tensor<(Seq, HeadDim), _, _> = self.sin.clone().gather(idx).realize();

        let neg_half_x: Tensor<(Seq, Headers, HeadDim), _, _> = (first_half.negate(), second_half)
            .concat_tensor_along(Axis::<2>)
            .realize();

        let y = sub_sin.broadcast_like(&x) * neg_half_x + sub_cos.broadcast_like(&x) * x;

        Ok(y)
    }

    //with tape
    pub fn try_forward_batch<Batch: Dim, Seq: Dim, Headers: Dim>(
        &self,
        x: Tensor<(Batch, Seq, Headers, HeadDim), E, D, OwnedTape<E, D>>,
    ) -> Result<Tensor<(Batch, Seq, Headers, HeadDim), E, D, OwnedTape<E, D>>, Error> {
        let seq = x.shape().1;
        let half_hiden = self.head_dim.size() / 2;
        let first_half = x.with_empty_tape().slice((.., .., .., 0..half_hiden));
        let second_half = x.with_empty_tape().slice((.., .., .., half_hiden..));

        let sub_cos: Tensor<(Seq, HeadDim), _, _, _> =
            self.cos.leaky_trace().slice((..seq.size(), ..)).realize();

        let sub_sin: Tensor<(Seq, HeadDim), _, _, _> =
            self.sin.leaky_trace().slice((..seq.size(), ..)).realize();

        let neg_half_x: Tensor<(Batch, Seq, Headers, HeadDim), _, _, _> =
            (first_half.negate(), second_half)
                .concat_tensor_along(Axis::<3>)
                .realize();

        let y = sub_sin.broadcast_like(&x) * neg_half_x + sub_cos.broadcast_like(&x) * x;

        Ok(y)
    }
}

pub fn new<HeadDim: Dim, E: Dtype, D: Device<E>>(
    head_dim: HeadDim,
    max_seq: usize,
    base: i64,
    dev: &D,
) -> RotaryEmbedding<HeadDim, E, D> {
    let half_hiden = head_dim.size() / 2;
    let theda = dev.tensor_from_vec(
        (0..head_dim.size())
            .step_by(2)
            .map(|c| E::from_usize(c).unwrap())
            .collect(),
        (half_hiden,),
    );
    let theda = ((theda / head_dim.size() as f32) * (base as f32).ln())
        .exp()
        .recip();

    let idx_theda: Tensor<(usize, usize), E, D> = (0..max_seq)
        .map(|i| theda.clone() * i as f32)
        .collect::<Vec<_>>()
        .stack()
        .realize();

    let idx_theda = (idx_theda.clone(), idx_theda)
        .concat_tensor_along(Axis::<1>)
        .realize();

    let cos = idx_theda.clone().cos();
    let sin = idx_theda.sin();

    RotaryEmbedding { head_dim, cos, sin }
}
