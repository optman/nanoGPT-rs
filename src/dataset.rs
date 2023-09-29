use std::fs::File;
use std::io::Read;
use std::path::Path;

use dfdx::data::ExactSizeDataset;

pub struct DataSet<const SEQ: usize> {
    data: Vec<u8>,
}

impl<const SEQ: usize> DataSet<SEQ> {
    pub fn new(path: &Path) -> Self {
        let mut data = Vec::<u8>::new();
        File::open(path).unwrap().read_to_end(&mut data).unwrap();

        Self { data }
    }
}

impl<const SEQ: usize> ExactSizeDataset for DataSet<SEQ> {
    type Item<'a> = (Vec<u8>, Vec<u8>) where Self: 'a;

    fn get(&self, index: usize) -> Self::Item<'_> {
        let mut x: Vec<u8> = Vec::with_capacity(SEQ);
        let mut y: Vec<u8> = Vec::with_capacity(SEQ);

        let mut start = (SEQ + 1) * index;
        x.extend(self.data[start..start + SEQ].iter());

        start += 1;
        y.extend(self.data[start..start + SEQ].iter());

        (x, y)
    }

    fn len(&self) -> usize {
        self.data.len() / (SEQ + 1)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_load_data() {
        DataSet::<5>::new(Path::new("input.txt"));
    }
}
