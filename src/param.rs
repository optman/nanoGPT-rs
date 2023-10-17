use dfdx::tensor::AutoDevice;

//model param
pub const LAYERS: usize = 6; /*12 */
pub const VOCAB: usize = 32000; //256;
pub const HIDEN: usize = 384/*768*/;
pub const MLP_HIDEN: usize = HIDEN * 4;
pub const HEADERS: usize = 12;
pub const HEADER_DIM: usize = HIDEN / HEADERS;
pub const MAX_SEQ: usize = 1024;

pub type E = f32;
pub type D = AutoDevice;
