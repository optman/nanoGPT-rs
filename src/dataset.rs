use std::fs::File;
use std::io::Read;
use std::path::Path;

use dfdx::data::ExactSizeDataset;

pub struct DataSet {
    ids: Vec<usize>,
    seq_len: usize,
}

impl DataSet {
    pub fn new(path: &Path, seq_len: usize) -> Self {
        let mut buf = Vec::<u8>::new();
        File::open(path).unwrap().read_to_end(&mut buf).unwrap();

        let ids = buf
            .chunks_exact(2)
            .map(|chunk| u16::from_be_bytes(chunk.try_into().unwrap()) as usize)
            .collect();

        Self { ids, seq_len }
    }
}

impl ExactSizeDataset for DataSet {
    type Item<'a> = (&'a[usize], &'a[usize]) where Self: 'a;

    fn get(&self, index: usize) -> Self::Item<'_> {
        let seq_len = self.seq_len;

        let mut start = (seq_len + 1) * index;
        let x = &self.ids[start..start + seq_len];

        start += 1;
        let y = &self.ids[start..start + seq_len];

        (x, y)
    }

    fn len(&self) -> usize {
        self.ids.len() / (self.seq_len + 1)
    }
}
