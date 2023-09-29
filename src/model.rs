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

const LAYERS: usize = 6; /*12 */
const VOCAB: usize = 256;
const HIDEN: usize = 384/*768*/;
const MLP_HIDEN: usize = HIDEN * 4;
const HEADERS: usize = 12;
const HEADER_DIM: usize = HIDEN / HEADERS;
const MAX_SEQ: usize = 1024;

pub struct GPTConfig {}

impl GPTConfig {
    pub fn init<E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform, D: Device<E>>(
        &self,
        dev: &D,
    ) -> GPTModel<E, D> {
        GPTModel::build(&dev)
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
impl<E, D: Device<E>> Module<Tensor<(usize, Const<HIDEN>), E, D>> for CausalSelfAttension<E, D>
where
    E: Dtype + num_traits::Float,
{
    type Output = Tensor<(usize, Const<HIDEN>), E, D>;
    type Error = D::Err;

    fn try_forward(
        &self,
        x: Tensor<(usize, Const<HIDEN>), E, D>,
    ) -> Result<Self::Output, Self::Error> {
        let dev = x.dev().clone();
        let (q, k, v) = self.attn.try_forward(x)?;

        let (seq, _hidden) = q.shape().concrete().into();
        let s: (usize, Const<HEADERS>, Const<HEADER_DIM>) = (seq, Const, Const);
        let q = q.try_reshape_like(&s)?.permute::<_, Axes3<1, 0, 2>>();
        let k = k.try_reshape_like(&s)?.permute::<_, Axes3<1, 0, 2>>();
        let v = v.try_reshape_like(&s)?.permute::<_, Axes3<1, 0, 2>>();

        let scale = (HEADER_DIM as f64).sqrt().recip();
        let att: Tensor<(Const<HEADERS>, usize, usize), _, _, _> =
            q.matmul(k.permute::<_, Axes3<0, 2, 1>>()) * scale;

        let s: (usize, usize) = (seq, seq);
        let mask = dev.upper_tri_like(&s, E::min_value(), 1);
        let att = mask.broadcast_like(&att) + att;

        let att = att.softmax::<Axis<2>>();
        let att = att
            .matmul(v)
            .permute::<_, Axes3<1, 0, 2>>()
            //.contiguous()
            .try_reshape_like(&(seq, Const))?;

        self.proj.try_forward(att)
    }
}
impl<const SEQ: usize, E: Dtype, D: Device<E>>
    ModuleMut<Tensor<(usize, Const<SEQ>, Const<HIDEN>), E, D, OwnedTape<E, D>>>
    for CausalSelfAttension<E, D>
where
    E: Dtype + num_traits::Float,
{
    type Output = Tensor<(usize, Const<SEQ>, Const<HIDEN>), E, D, OwnedTape<E, D>>;
    type Error = D::Err;

    fn try_forward_mut(
        &mut self,
        x: Tensor<(usize, Const<SEQ>, Const<HIDEN>), E, D, OwnedTape<E, D>>,
    ) -> Result<Self::Output, Self::Error> {
        let dev = x.dev().clone();
        let (q, k, v) = self.attn.try_forward_mut(x)?;

        let (batch, _seq, _hidden) = q.shape().concrete().into();
        let s: (usize, Const<SEQ>, Const<HEADERS>, Const<HEADER_DIM>) =
            (batch, Const, Const, Const);
        let q = q.try_reshape_like(&s)?.permute::<_, Axes4<0, 2, 1, 3>>();
        let k = k.try_reshape_like(&s)?.permute::<_, Axes4<0, 2, 1, 3>>();
        let v = v.try_reshape_like(&s)?.permute::<_, Axes4<0, 2, 1, 3>>();

        let scale = (HEADER_DIM as f64).sqrt().recip();
        let att: Tensor<(usize, Const<HEADERS>, Const<SEQ>, Const<SEQ>), _, _, _> =
            q.matmul(k.permute::<_, Axes4<0, 1, 3, 2>>()) * scale;

        let s: Rank2<SEQ, SEQ> = (Const, Const);
        let mask = dev.upper_tri_like(&s, E::min_value(), 1);
        let att = mask.broadcast_like(&att).leaky_traced() + att;

        let att = att.softmax::<Axis<3>>();
        let att = att
            .matmul(v)
            .permute::<_, Axes4<0, 2, 1, 3>>()
            //.contiguous()
            .try_reshape_like(&(batch, Const, Const))?;

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

impl<E, D: Device<E>> Module<Tensor<(usize, Const<HIDEN>), E, D>> for Block<E, D>
where
    E: Dtype + num_traits::Float,
{
    type Output = Tensor<(usize, Const<HIDEN>), E, D>;
    type Error = D::Err;

    fn try_forward(
        &self,
        x: Tensor<(usize, Const<HIDEN>), E, D>,
    ) -> Result<Self::Output, Self::Error> {
        let x = self.attn.try_forward(x)?;
        self.mlp.try_forward(x)
    }
}

impl<const SEQ: usize, E: Dtype, D: Device<E>>
    ModuleMut<Tensor<(usize, Const<SEQ>, Const<HIDEN>), E, D, OwnedTape<E, D>>> for Block<E, D>
where
    E: Dtype + num_traits::Float,
{
    type Output = Tensor<(usize, Const<SEQ>, Const<HIDEN>), E, D, OwnedTape<E, D>>;
    type Error = D::Err;

    fn try_forward_mut(
        &mut self,
        x: Tensor<(usize, Const<SEQ>, Const<HIDEN>), E, D, OwnedTape<E, D>>,
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
                atten_layers: atten_layers,
                lm_header,
            },
        )
    }
}

impl<E: Dtype, D: Device<E>> Module<Tensor<(usize,), usize, D>> for GPTModel<E, D>
where
    E: Dtype + num_traits::Float,
{
    type Output = Tensor<(usize, Const<VOCAB>), E, D>;
    type Error = D::Err;

    fn try_forward(&self, x: Tensor<(usize,), usize, D>) -> Result<Self::Output, D::Err> {
        let seq_len = x.shape().0;
        let dst: (usize,) = (seq_len,);
        let pos = x.dev().tensor_from_vec((0..seq_len).collect(), dst);
        let pos = self.pos_enc.try_forward(pos)?;

        let x = self.embedding_layer.try_forward(x)?;
        let x = pos + x;

        let x = self.atten_layers.try_forward(x)?;
        self.lm_header.try_forward(x)
    }
}

impl<const SEQ: usize, E: Dtype, D: Device<E>>
    ModuleMut<Tensor<(usize, Const<SEQ>), usize, D, OwnedTape<E, D>>> for GPTModel<E, D>
where
    E: Dtype + num_traits::Float,
{
    type Output = Tensor<(usize, Const<SEQ>, Const<VOCAB>), E, D, OwnedTape<E, D>>;
    type Error = D::Err;

    fn try_forward_mut(
        &mut self,
        x: Tensor<(usize, Const<SEQ>), usize, D, OwnedTape<E, D>>,
    ) -> Result<Self::Output, D::Err> {
        let dst: Rank1<SEQ> = (Const,);
        let pos = x.dev().tensor_from_vec((0..SEQ).collect(), dst);
        let pos = self.pos_enc.try_forward_mut(pos)?;

        let x = self.embedding_layer.try_forward_mut(x)?;
        let x = pos.broadcast_like(&x).leaky_traced() + x;

        let x = self.atten_layers.try_forward_mut(x)?;
        self.lm_header.try_forward_mut(x)
    }
}
