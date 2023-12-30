use crate::{
    rotary::{RotaryEmbedding, RotaryEmbeddingConfig},
    Cache,
};
use dfdx::{
    nn::{FastGeLU, Module},
    prelude::*,
    tensor::{Error, Tensor},
};
use std::fmt::Debug;

#[derive(Debug, Clone, CustomModule)]
#[built(CausalSelfAttension)]
pub struct CausalSelfAttensionConfig<P: Params> {
    #[module]
    attn: SplitInto<(
        LinearConfig<P::Hidden, P::Hidden>,
        LinearConfig<P::Hidden, P::KvDim>,
        LinearConfig<P::Hidden, P::KvDim>,
    )>,
    #[module]
    attn_dropout: Dropout,
    #[module]
    proj: LinearConfig<P::Hidden, P::Hidden>,
    #[module]
    proj_dropout: Dropout,
    p: P,
}

impl<P: Params> CausalSelfAttensionConfig<P> {
    pub fn new(p: P) -> Self {
        Self {
            attn: SplitInto((
                LinearConfig {
                    inp: p.hidden(),
                    out: p.hidden(),
                },
                LinearConfig {
                    inp: p.hidden(),
                    out: p.kv_dim(),
                },
                LinearConfig {
                    inp: p.hidden(),
                    out: p.kv_dim(),
                },
            )),
            attn_dropout: Dropout { p: p.dropout() },
            proj: LinearConfig {
                inp: p.hidden(),
                out: p.hidden(),
            },
            proj_dropout: Dropout { p: p.dropout() },
            p,
        }
    }
}

impl<P: Params, E, D: Device<E>> CausalSelfAttension<P, E, D>
where
    E: Dtype + num_traits::Float,
{
    fn try_forward<Seq: Dim>(
        &self,
        x: Tensor<(Seq, P::Hidden), E, D>,
        layer: usize,
        pos: usize,
        pos_scale: usize,
        pos_enc: &RotaryEmbedding<P::HeadDim, E, D>,
        mut cache: Option<Cache<P::KvHeads, P::HeadDim, P::Layers, E, D>>,
    ) -> Result<
        (
            Tensor<(Seq, P::Hidden), E, D>,
            Option<Cache<P::KvHeads, P::HeadDim, P::Layers, E, D>>,
        ),
        Error,
    > {
        let dev = x.dev().clone();
        let hidden = x.shape().1;
        let (q, k, v) = self.attn.try_forward(x)?;
        let (seq, _hidden) = q.shape().concrete().into();

        let qs = (seq, self.p.heads(), self.p.head_dim());
        let q = q.try_reshape_like(&qs)?;
        let q = pos_enc.try_forward(q, pos, pos_scale)?;

        let kvs = (seq, self.p.kv_heads(), self.p.head_dim());
        let k = k.reshape_like(&kvs);
        let mut k = pos_enc.try_forward(k, pos, pos_scale)?;

        let mut v = v.reshape_like(&kvs);

        if let Some(cache) = cache.as_mut() {
            (k, v) = cache.append(layer, k, v);
        }

        //(seq, header, header_dim) -> (headers, seq, header_dim)
        let q = q.permute::<_, Axes3<1, 0, 2>>();

        let repeate = self.p.heads().size() / self.p.kv_heads().size();
        let (kv_seq, _kv_headers, _hidden) = k.shape().concrete().into();
        let kvs2 = (kv_seq, self.p.kv_heads(), repeate, self.p.head_dim());
        let kvs3 = (kv_seq, self.p.heads(), self.p.head_dim());

        let k = k
            .broadcast_like(&kvs2)
            .try_reshape_like(&kvs3)?
            .permute::<_, Axes3<1, 0, 2>>();

        let v = v
            .broadcast_like(&kvs2)
            .try_reshape_like(&kvs3)?
            .permute::<_, Axes3<1, 0, 2>>();

        let scale = (self.p.head_dim().size() as f32).sqrt().recip();
        let att: Tensor<(P::Heads, usize, usize), _, _, _> =
            q.matmul(k.permute::<_, Axes3<0, 2, 1>>()) * scale;

        let attn_seq = kv_seq;
        let mask = dev.upper_tri_like(&(attn_seq, attn_seq), E::neg_infinity(), 1);
        let sub_mask_sel = ((attn_seq - seq)..attn_seq).collect();
        let sub_mask_sel = dev.tensor_from_vec(sub_mask_sel, (seq,));
        let mask = mask.gather(sub_mask_sel);
        let att = mask.broadcast_like(&att) + att;

        let att = att.softmax::<Axis<2>>();

        let att: Tensor<(Seq, P::Hidden), E, D> = att
            .matmul(v)
            .permute::<_, Axes3<1, 0, 2>>()
            //.contiguous()
            .try_reshape_like(&(seq, hidden))?
            .realize();

        Ok((self.proj.try_forward(att)?, cache))
    }
}

impl<P: Params, E: Dtype, D: Device<E>> CausalSelfAttension<P, E, D>
where
    E: Dtype + num_traits::Float,
{
    fn try_forward_mut<Batch: Dim, Seq: Dim>(
        &mut self,
        x: Tensor<(Batch, Seq, P::Hidden), E, D, OwnedTape<E, D>>,
        pos_enc: &RotaryEmbedding<P::HeadDim, E, D>,
    ) -> Result<Tensor<(Batch, Seq, P::Hidden), E, D, OwnedTape<E, D>>, Error> {
        let dev = x.dev().clone();
        let hidden = x.shape().2;
        let (q, k, v) = self.attn.try_forward_mut(x)?;

        let (batch, seq, _hidden) = q.shape().concrete().into();
        let qs = (batch, seq, self.p.heads(), self.p.head_dim());
        let q = q.try_reshape_like(&qs)?;
        let q = pos_enc.try_forward_batch(q)?;

        let kvs = (batch, seq, self.p.kv_heads(), self.p.head_dim());
        let k = k.try_reshape_like(&kvs)?;
        let v = v.try_reshape_like(&kvs)?;
        let k = pos_enc.try_forward_batch(k)?;

        //(batch,seq, headers, header_dim) -> (batch, header, seq, header_dim)
        let q = q.permute::<_, Axes4<0, 2, 1, 3>>();

        let repeate = self.p.heads().size() / self.p.kv_heads().size();
        let kvs2 = (batch, seq, self.p.kv_heads(), repeate, self.p.head_dim());

        let k = k
            .broadcast_like(&kvs2)
            .try_reshape_like(&qs)?
            .permute::<_, Axes4<0, 2, 1, 3>>();

        let v = v
            .broadcast_like(&kvs2)
            .try_reshape_like(&qs)?
            .permute::<_, Axes4<0, 2, 1, 3>>();

        let scale = (self.p.head_dim().size() as f64).sqrt().recip();
        let att: Tensor<(usize, P::Heads, usize, usize), _, _, _> =
            q.matmul(k.permute::<_, Axes4<0, 1, 3, 2>>()) * scale;

        let mask = dev.upper_tri_like(&(seq, seq), E::neg_infinity(), 1);
        let att = mask.broadcast_like::<_, Axes2<0, 1>>(&att).leaky_traced() + att;
        let att = att.softmax::<Axis<3>>();
        let att = self.attn_dropout.try_forward_mut(att)?;
        let att: Tensor<(Batch, Seq, P::Hidden), _, _, _> = att
            .matmul(v)
            .permute::<_, Axes4<0, 2, 1, 3>>()
            //.contiguous()
            .try_reshape_like(&(batch, seq, hidden))?
            .realize();

        let y = self.proj.try_forward_mut(att)?;
        self.proj_dropout.try_forward_mut(y)
    }
}

#[derive(Debug, Clone, CustomModule)]
#[built(Block)]
pub struct BlockConfig<P: Params> {
    #[module]
    norm: LayerNorm1DConfig<P::Hidden>,
    #[module]
    attn: CausalSelfAttensionConfig<P>,
    #[module]
    mlp: ResidualAdd<(
        LayerNorm1DConfig<P::Hidden>,
        (
            LinearConfig<P::Hidden, P::MlpHidden>,
            FastGeLU,
            LinearConfig<P::MlpHidden, P::Hidden>,
        ),
        Dropout,
    )>,
}

impl<P: Params> BlockConfig<P> {
    pub fn new(p: P) -> Self {
        Self {
            norm: LayerNorm1DConfig(p.hidden()),
            attn: CausalSelfAttensionConfig::new(p),
            mlp: ResidualAdd((
                LayerNorm1DConfig(p.hidden()),
                (
                    LinearConfig {
                        inp: p.hidden(),
                        out: p.mlp_hidden(),
                    },
                    FastGeLU,
                    LinearConfig {
                        inp: p.mlp_hidden(),
                        out: p.hidden(),
                    },
                ),
                Dropout { p: p.dropout() },
            )),
        }
    }
}

impl<P: Params, E, D: Device<E>> Block<P, E, D>
where
    E: Dtype + num_traits::Float,
{
    pub fn try_forward<Seq: Dim>(
        &self,
        x: Tensor<(Seq, P::Hidden), E, D>,
        layer: usize,
        pos: usize,
        pos_scale: usize,
        pos_enc: &RotaryEmbedding<P::HeadDim, E, D>,
        cache: Option<Cache<P::KvHeads, P::HeadDim, P::Layers, E, D>>,
    ) -> Result<
        (
            Tensor<(Seq, P::Hidden), E, D>,
            Option<Cache<P::KvHeads, P::HeadDim, P::Layers, E, D>>,
        ),
        Error,
    > {
        let x2 = self.norm.try_forward(x.clone())?;
        let (x2, cache) = self
            .attn
            .try_forward(x2, layer, pos, pos_scale, pos_enc, cache)?;
        let x = x2.try_add(x)?;
        Ok((self.mlp.try_forward(x)?, cache))
    }

    fn try_forward_mut<Batch: Dim, Seq: Dim>(
        &mut self,
        x: Tensor<(Batch, Seq, P::Hidden), E, D, OwnedTape<E, D>>,
        pos_enc: &RotaryEmbedding<P::HeadDim, E, D>,
    ) -> Result<Tensor<(Batch, Seq, P::Hidden), E, D, OwnedTape<E, D>>, Error> {
        let x2 = self.norm.try_forward_mut(x.with_empty_tape())?;
        let x2 = self.attn.try_forward_mut(x2, pos_enc)?;
        let x = x2.try_add(x)?;
        self.mlp.try_forward_mut(x)
    }
}

#[derive(Clone, Debug, CustomModule)]
#[built(GPTModel)]
pub struct GPTModelConfig<P: Params> {
    #[module]
    pub embedding_layer: EmbeddingConfig<P::Vocab, P::Hidden>,
    #[module]
    pub pos_enc: RotaryEmbeddingConfig<P::HeadDim>,
    #[module]
    pub dropout: Dropout,
    #[module]
    pub atten_layers: Vec<BlockConfig<P>>,
    #[module]
    pub lm_header: LinearConfig<P::Hidden, P::Vocab>,
    pub params: P,
}

impl<P: Params, E: Dtype, D: Device<E>> GPTModel<P, E, D>
where
    E: Dtype + num_traits::Float,
    D: Device<f64>,
{
    pub fn params(&self) -> &P {
        &self.params
    }

    fn layers(&self) -> usize {
        self.params.layers().size()
    }

    pub fn try_forward<Seq: Dim>(
        &self,
        x: Tensor<(Seq,), usize, D>,
        pos: usize,
        pos_scale: usize,
        cache: Option<Cache<P::KvHeads, P::HeadDim, P::Layers, E, D>>,
    ) -> Result<
        (
            Tensor<(Seq, P::Vocab), E, D>,
            Option<Cache<P::KvHeads, P::HeadDim, P::Layers, E, D>>,
        ),
        Error,
    > {
        let mut x = self.embedding_layer.try_forward(x)?;
        let mut cache = cache;
        for i in 0..self.layers() {
            (x, cache) =
                self.atten_layers[i].try_forward(x, i, pos, pos_scale, &self.pos_enc, cache)?;
        }
        Ok((self.lm_header.try_forward(x)?, cache))
    }

    pub fn try_forward_mut<Batch: Dim, Seq: Dim>(
        &mut self,
        x: Tensor<(Batch, Seq), usize, D, OwnedTape<E, D>>,
    ) -> Result<Tensor<(Batch, Seq, P::Vocab), E, D, OwnedTape<E, D>>, Error> {
        let x = self.embedding_layer.try_forward_mut(x)?;
        let mut x = self.dropout.try_forward_mut(x)?;

        for i in 0..self.layers() {
            x = self.atten_layers[i].try_forward_mut(x, &self.pos_enc)?;
        }

        self.lm_header.try_forward_mut(x)
    }
}

pub trait Params: Debug + Clone + Copy {
    type Vocab: Dim;
    type Hidden: Dim;
    type MlpHidden: Dim;
    type Heads: Dim;
    type HeadDim: Dim;
    type KvHeads: Dim;
    type KvDim: Dim;
    type Layers: Dim;

    fn vocab(&self) -> Self::Vocab;
    fn hidden(&self) -> Self::Hidden;
    fn mlp_hidden(&self) -> Self::MlpHidden;
    fn heads(&self) -> Self::Heads;
    fn head_dim(&self) -> Self::HeadDim;
    fn kv_heads(&self) -> Self::KvHeads;
    fn kv_dim(&self) -> Self::KvDim;
    fn layers(&self) -> Self::Layers;
    fn dropout(&self) -> f64;
}
