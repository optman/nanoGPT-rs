use crate::{param::*, Cache};
use dfdx::{
    nn::modules::{
        Embedding, FastGeLU, LayerNorm1D, Linear, Module, ModuleVisitor, Repeated, Residual,
        TensorCollection,
    },
    prelude::BuildModule,
    prelude::*,
    tensor::Tensor,
    tensor_ops::Device,
};

pub struct GPTConfig {}

impl GPTConfig {
    pub fn init<E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform, D: Device<E>>(
        &self,
        dev: &D,
    ) -> GPTModel<E, D> {
        GPTModel::build(dev)
    }
}

pub struct CausalSelfAttension<E: Dtype, D: Device<E>> {
    attn: SplitInto<(
        Linear<HIDEN, HIDEN, E, D>,
        Linear<HIDEN, HIDEN, E, D>,
        Linear<HIDEN, HIDEN, E, D>,
    )>,
    proj: Linear<HIDEN, HIDEN, E, D>,
}
impl<E: Dtype, D: Device<E>> TensorCollection<E, D> for CausalSelfAttension<E, D>
where
    E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
{
    type To<E2: Dtype, D2: Device<E2>> = CausalSelfAttension<E2, D2>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module("attn", |s| &s.attn, |s| &mut s.attn),
                Self::module("porj", |s| &s.proj, |s| &mut s.proj),
            ),
            |(attn, proj)| CausalSelfAttension { attn, proj },
        )
    }
}
impl<E, D: Device<E>> CausalSelfAttension<E, D>
where
    E: Dtype + num_traits::Float,
{
    fn try_forward<Seq: Dim>(
        &self,
        x: Tensor<(Seq, Const<HIDEN>), E, D>,
        layer: usize,
        mut cache: Option<Cache<HEADERS, HEADER_DIM, LAYERS, E, D>>,
    ) -> Result<
        (
            Tensor<(Seq, Const<HIDEN>), E, D>,
            Option<Cache<HEADERS, HEADER_DIM, LAYERS, E, D>>,
        ),
        D::Err,
    > {
        let dev = x.dev().clone();
        let (q, k, v) = self.attn.try_forward(x)?;

        let (seq, _hidden) = q.shape().concrete().into();
        let s = (seq, Const::<HEADERS>, Const::<HEADER_DIM>);
        let q = q.try_reshape_like(&s)?.permute::<_, Axes3<1, 0, 2>>();
        let mut k = k.try_reshape_like(&s)?.permute::<_, Axes3<1, 0, 2>>();
        let mut v = v.try_reshape_like(&s)?.permute::<_, Axes3<1, 0, 2>>();

        if let Some(cache) = cache.as_mut() {
            if let Some(c_k) = &cache.k[layer] {
                k = (c_k.clone(), k).concat_along(Axis::<1>)
            }
            cache.k[layer] = Some(k.clone());

            if let Some(c_v) = &cache.v[layer] {
                v = (c_v.clone(), v).concat_along(Axis::<1>)
            }
            cache.v[layer] = Some(v.clone());
        }

        let scale = (HEADER_DIM as f64).sqrt().recip();
        let att: Tensor<(Const<HEADERS>, usize, usize), _, _, _> =
            q.matmul(k.permute::<_, Axes3<0, 2, 1>>()) * scale;

        let attn_seq: usize = v.shape().1.size();
        let mask = dev.upper_tri_like(&(attn_seq, attn_seq), E::min_value(), 1);
        let sub_mask_sel = ((attn_seq - seq)..attn_seq).collect();
        let sub_mask_sel = dev.tensor_from_vec(sub_mask_sel, (seq,));
        let mask = mask.gather(sub_mask_sel);
        let att = mask.broadcast_like(&att) + att;

        let att = att.softmax::<Axis<2>>();
        let att: Tensor<(Seq, Const<HIDEN>), E, D> = att
            .matmul(v)
            .permute::<_, Axes3<1, 0, 2>>()
            //.contiguous()
            .try_reshape_like(&(seq, Const::<HIDEN>))?
            .realize();

        Ok((self.proj.try_forward(att)?, cache))
    }
}

impl<Batch: Dim, Seq: Dim, E: Dtype, D: Device<E>>
    ModuleMut<Tensor<(Batch, Seq, Const<HIDEN>), E, D, OwnedTape<E, D>>>
    for CausalSelfAttension<E, D>
where
    E: Dtype + num_traits::Float,
{
    type Output = Tensor<(Batch, Seq, Const<HIDEN>), E, D, OwnedTape<E, D>>;
    type Error = D::Err;

    fn try_forward_mut(
        &mut self,
        x: Tensor<(Batch, Seq, Const<HIDEN>), E, D, OwnedTape<E, D>>,
    ) -> Result<Self::Output, Self::Error> {
        let dev = x.dev().clone();
        let (q, k, v) = self.attn.try_forward_mut(x)?;

        let (batch, seq, _hidden) = q.shape().concrete().into();
        let s = (batch, seq, Const::<HEADERS>, Const::<HEADER_DIM>);
        let q = q.try_reshape_like(&s)?.permute::<_, Axes4<0, 2, 1, 3>>();
        let k = k.try_reshape_like(&s)?.permute::<_, Axes4<0, 2, 1, 3>>();
        let v = v.try_reshape_like(&s)?.permute::<_, Axes4<0, 2, 1, 3>>();

        let scale = (HEADER_DIM as f64).sqrt().recip();
        let att: Tensor<(usize, Const<HEADERS>, usize, usize), _, _, _> =
            q.matmul(k.permute::<_, Axes4<0, 1, 3, 2>>()) * scale;

        let mask = dev.upper_tri_like(&(seq, seq), E::min_value(), 1);
        let att = mask.broadcast_like::<_, Axes2<0, 1>>(&att).leaky_traced() + att;
        let att = att.softmax::<Axis<3>>();
        let att: Tensor<(Batch, Seq, Const<HIDEN>), _, _, _> = att
            .matmul(v)
            .permute::<_, Axes4<0, 2, 1, 3>>()
            //.contiguous()
            .try_reshape_like(&(batch, seq, Const::<HIDEN>))?
            .realize();

        self.proj.try_forward_mut(att)
    }
}

pub struct Block<E: Dtype, D: Device<E>> {
    attn: Residual<(LayerNorm1D<HIDEN, E, D>, CausalSelfAttension<E, D>)>,
    mlp: Residual<(
        LayerNorm1D<HIDEN, E, D>,
        (
            Linear<HIDEN, MLP_HIDEN, E, D>,
            FastGeLU,
            Linear<MLP_HIDEN, HIDEN, E, D>,
        ),
    )>,
}

impl<E: Dtype, D: Device<E>> TensorCollection<E, D> for Block<E, D>
where
    E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
{
    type To<E2: Dtype, D2: Device<E2>> = Block<E2, D2>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module("attn", |s| &s.attn, |s| &mut s.attn),
                Self::module("mlp", |s| &s.mlp, |s| &mut s.mlp),
            ),
            |(attn, mlp)| Block { attn, mlp },
        )
    }
}

impl<E, D: Device<E>> Block<E, D>
where
    E: Dtype + num_traits::Float,
{
    pub fn try_forward<Seq: Dim>(
        &self,
        x: Tensor<(Seq, Const<HIDEN>), E, D>,
        layer: usize,
        cache: Option<Cache<HEADERS, HEADER_DIM, LAYERS, E, D>>,
    ) -> Result<
        (
            Tensor<(Seq, Const<HIDEN>), E, D>,
            Option<Cache<HEADERS, HEADER_DIM, LAYERS, E, D>>,
        ),
        D::Err,
    > {
        let (norm, attn) = &self.attn.0;
        let x2 = norm.try_forward(x.clone())?;
        let (x2, cache) = attn.try_forward(x2, layer, cache)?;
        let x = x2.try_add(x)?;
        Ok((self.mlp.try_forward(x)?, cache))
    }
}

impl<Batch: Dim, Seq: Dim, E: Dtype, D: Device<E>>
    ModuleMut<Tensor<(Batch, Seq, Const<HIDEN>), E, D, OwnedTape<E, D>>> for Block<E, D>
where
    E: Dtype + num_traits::Float,
{
    type Output = Tensor<(Batch, Seq, Const<HIDEN>), E, D, OwnedTape<E, D>>;
    type Error = D::Err;

    fn try_forward_mut(
        &mut self,
        x: Tensor<(Batch, Seq, Const<HIDEN>), E, D, OwnedTape<E, D>>,
    ) -> Result<Self::Output, Self::Error> {
        let x = self.attn.try_forward_mut(x)?;
        self.mlp.try_forward_mut(x)
    }
}

pub struct GPTModel<E: Dtype, D: Device<E>> {
    embedding_layer: Embedding<VOCAB, HIDEN, E, D>,
    pos_enc: Embedding<MAX_SEQ, HIDEN, E, D>,
    atten_layers: Repeated<Block<E, D>, LAYERS>,
    lm_header: Linear<HIDEN, VOCAB, E, D>,
}

impl<E: Dtype, D: Device<E>> TensorCollection<E, D> for GPTModel<E, D>
where
    E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
{
    type To<E2: Dtype, D2: Device<E2>> = GPTModel<E2, D2>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module(
                    "embedding",
                    |s| &s.embedding_layer,
                    |s| &mut s.embedding_layer,
                ),
                Self::module("pos_enc", |s| &s.pos_enc, |s| &mut s.pos_enc),
                Self::module("atten_layers", |s| &s.atten_layers, |s| &mut s.atten_layers),
                Self::module("lm_header", |s| &s.lm_header, |s| &mut s.lm_header),
            ),
            |(embedding_layer, pos_enc, atten_layers, lm_header)| GPTModel {
                embedding_layer,
                pos_enc,
                atten_layers,
                lm_header,
            },
        )
    }
}

impl<E: Dtype, D: Device<E>> GPTModel<E, D>
where
    E: Dtype + num_traits::Float,
{
    pub fn try_forward<Seq: Dim>(
        &self,
        x: Tensor<(Seq,), usize, D>,
        pos: usize,
        cache: Option<Cache<HEADERS, HEADER_DIM, LAYERS, E, D>>,
    ) -> Result<
        (
            Tensor<(Seq, Const<VOCAB>), E, D>,
            Option<Cache<HEADERS, HEADER_DIM, LAYERS, E, D>>,
        ),
        D::Err,
    > {
        let seq_len = x.shape().0.size();
        let pos = x
            .dev()
            .tensor_from_vec((pos..pos + seq_len).collect(), (seq_len,));
        let pos = self.pos_enc.try_forward(pos)?;

        let x = self.embedding_layer.try_forward(x)?;
        let mut x = pos.realize() + x;

        let mut cache = cache;
        for i in 0..LAYERS {
            (x, cache) = self.atten_layers[i].try_forward(x, i, cache)?;
        }
        Ok((self.lm_header.try_forward(x)?, cache))
    }
}

impl<Batch: Dim, Seq: Dim, E: Dtype, D: Device<E>>
    ModuleMut<Tensor<(Batch, Seq), usize, D, OwnedTape<E, D>>> for GPTModel<E, D>
where
    E: Dtype + num_traits::Float,
{
    type Output = Tensor<(Batch, Seq, Const<VOCAB>), E, D, OwnedTape<E, D>>;
    type Error = D::Err;

    fn try_forward_mut(
        &mut self,
        x: Tensor<(Batch, Seq), usize, D, OwnedTape<E, D>>,
    ) -> Result<Self::Output, D::Err> {
        let seq = x.shape().1;
        let pos = x.dev().tensor_from_vec((0..seq.size()).collect(), (seq,));
        let pos = self.pos_enc.try_forward_mut(pos)?;

        let x = self.embedding_layer.try_forward_mut(x)?;
        let x = pos.broadcast_like(&x).leaky_traced() + x;

        let x = self.atten_layers.try_forward_mut(x)?;
        self.lm_header.try_forward_mut(x)
    }
}
