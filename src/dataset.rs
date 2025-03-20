use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use dfdx::data::ExactSizeDataset;

pub struct DataSet {
    ids: Vec<Vec<usize>>,
}

impl DataSet {
    pub fn new(path: &Path) -> Self {
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);
        let mut ids = Vec::new();

        for line in reader.lines() {
            let line = line.unwrap();
            let line_ids: Vec<usize> = line.bytes().map(|b| b as usize).collect();

            ids.push(line_ids);
        }
        Self { ids }
    }
}

impl ExactSizeDataset for DataSet {
    type Item<'a> = &'a[usize] where Self: 'a;

    fn get(&self, index: usize) -> Self::Item<'_> {
        &self.ids[index]
    }

    fn len(&self) -> usize {
        self.ids.len()
    }
}
