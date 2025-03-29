pub struct Map2<K, V> {
    free_keys: Vec<K>,
    values: Vec<Option<V>>,
}

impl<K: TryFrom<usize> + Into<usize> + Copy + std::fmt::Debug, V> Map2<K, V> {
    pub fn new() -> Self {
        Self {
            free_keys: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn insert(&mut self, value: V) -> Result<K, V> {
        if let Some(key) = self.free_keys.pop() {
            match std::mem::replace(&mut self.values[key.into()], Some(value)) {
                None => Ok(key),
                Some(value) => Err(value),
            }
        } else {
            self.values.push(Some(value));
            Ok(K::try_from(self.values.len() - 1).expect(""))
        }
    }

    pub fn get(&self, key: K) -> Option<&V> {
        self.values.get(key.into())?.as_ref()
    }

    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        self.values.get_mut(key.into())?.as_mut()
    }

    pub fn remove(&mut self, key: K) -> Option<V> {
        let out = std::mem::take(&mut self.values[key.into()]);
        if out.is_some() {
            self.free_keys.push(key);
        }
        out
    }
}
