//! Byte-exact port of MRI 3.4's `Enumerable#sort_by` for primitive keys
//! (`enum.c`: `rb_uniform_intro_sort_2` and friends).
//!
//! The gem's strategies rank candidates with `sort_by` over `Float` scores
//! or `Integer` distances. MRI sorts those with a uniform introsort whose
//! tie order — quicksort pivoting by *value*, insertion sort below 17
//! elements, heapsort at depth exhaustion — is deterministic but not stable.
//! The frozen conformance vectors encode that tie order, so this module
//! reproduces the algorithm exactly (indices are `isize` because the C
//! partition pointers may walk one slot past the sub-array ends).
//!
//! Mixed-type keys would instead take MRI's `ruby_qsort` path; the gem
//! always yields homogeneous keys on this path, so that variant is not
//! ported.

/// Sort `items` by `key`, reproducing MRI's uniform introsort tie order.
pub fn sort_by<T, K>(items: &mut [T], key: impl Fn(&T) -> K)
where
    K: Copy + PartialOrd,
{
    let n = items.len();
    if n <= 1 {
        // `enum_sort_by` only sorts when more than one pair was collected.
        return;
    }

    // Schwartzian transform: pairs of (key, original index).
    let mut pairs: Vec<(K, usize)> = items.iter().map(|item| (key(item), 0)).collect();
    for (i, (_, index)) in pairs.iter_mut().enumerate() {
        *index = i;
    }

    let sorted = pairs.windows(2).all(|w| !is_larger(&w[0].0, &w[1].0));
    if !sorted {
        // `d = CHAR_BIT * sizeof(n) - nlz(n) - 1`, doubled at the call site.
        let d = 2 * (usize::BITS as usize - n.leading_zeros() as usize - 1);
        quicksort_intro(&mut pairs, d);
    }

    // Apply the permutation without `T: Clone`.
    let mut permutation: Vec<usize> = pairs.iter().map(|pair| pair.1).collect();
    apply_permutation(items, &mut permutation);
}

fn is_larger<K: PartialOrd>(a: &K, b: &K) -> bool {
    a > b
}

fn is_less<K: PartialOrd>(a: &K, b: &K) -> bool {
    a < b
}

/// `rb_uniform_insertionsort_2`: stable insertion sort (strict `less`
/// comparisons; fast path when the element belongs at the front).
fn insertion_sort<K: Copy + PartialOrd>(pairs: &mut [(K, usize)]) {
    if pairs.len() < 2 {
        return;
    }
    for index in 1..pairs.len() {
        let tmp = pairs[index];
        if is_less(&tmp.0, &pairs[0].0) {
            pairs.copy_within(0..index, 1);
            pairs[0] = tmp;
        } else {
            let mut k = index;
            while k > 0 && is_less(&tmp.0, &pairs[k - 1].0) {
                pairs[k] = pairs[k - 1];
                k -= 1;
            }
            pairs[k] = tmp;
        }
    }
}

/// `rb_uniform_heap_down_2` — `len` is the last valid index, mirroring the
/// C signature.
fn heap_down<K: Copy + PartialOrd>(pairs: &mut [(K, usize)], offset: usize, len: usize) {
    let tmp = pairs[offset];
    let mut offset = offset;
    loop {
        let mut child = offset * 2 + 1;
        if child > len {
            break;
        }
        if child < len && is_less(&pairs[child].0, &pairs[child + 1].0) {
            child += 1;
        }
        if !is_less(&tmp.0, &pairs[child].0) {
            break;
        }
        pairs[offset] = pairs[child];
        offset = child;
    }
    pairs[offset] = tmp;
}

/// `rb_uniform_heapsort_2`.
fn heapsort<K: Copy + PartialOrd>(pairs: &mut [(K, usize)]) {
    let n = pairs.len();
    if n < 2 {
        return;
    }
    let mut offset = n >> 1;
    while offset > 0 {
        offset -= 1;
        heap_down(pairs, offset, n - 1);
    }
    let mut offset = n - 1;
    while offset > 0 {
        pairs.swap(0, offset);
        offset -= 1;
        heap_down(pairs, 0, offset);
    }
}

/// `med3_val` — median of three key *values*.
fn med3_val<K: Copy + PartialOrd>(a: K, b: K, c: K) -> K {
    if is_less(&a, &b) {
        if is_less(&b, &c) {
            b
        } else if is_less(&c, &a) {
            a
        } else {
            c
        }
    } else if is_less(&c, &b) {
        b
    } else if is_less(&a, &c) {
        a
    } else {
        c
    }
}

/// `rb_uniform_quicksort_intro_2` — Hoare partition by pivot value, right
/// side recursed first.
fn quicksort_intro<K: Copy + PartialOrd>(pairs: &mut [(K, usize)], d: usize) {
    let len = pairs.len();
    if len <= 16 {
        insertion_sort(pairs);
        return;
    }
    if d == 0 {
        heapsort(pairs);
        return;
    }

    let pivot = med3_val(pairs[0].0, pairs[len >> 1].0, pairs[len - 1].0);
    let mut i: isize = 0;
    let mut j: isize = len as isize - 1;
    loop {
        while is_less(&pairs[i as usize].0, &pivot) {
            i += 1;
        }
        while is_less(&pivot, &pairs[j as usize].0) {
            j -= 1;
        }
        if i <= j {
            pairs.swap(i as usize, j as usize);
            i += 1;
            j -= 1;
        }
        if i > j {
            break;
        }
    }
    let j = (j + 1) as usize;
    if len - j > 1 {
        quicksort_intro(&mut pairs[j..], d - 1);
    }
    if i as usize > 1 {
        quicksort_intro(&mut pairs[..i as usize], d - 1);
    }
}

/// Rotate `items` into `permutation` order without cloning: permutation
/// cycles are followed with `std::mem::replace`.
fn apply_permutation<T>(items: &mut [T], permutation: &mut [usize]) {
    for start in 0..items.len() {
        let mut current = start;
        while permutation[current] != start {
            let next = permutation[current];
            items.swap(current, next);
            permutation[current] = current;
            current = next;
        }
        permutation[current] = current;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tie permutations frozen from Ruby 3.4.8 (`Array#sort_by` with
    /// uniform primitive keys). If MRI's algorithm changes upstream, these
    /// fixtures — like the conformance vectors — move with a re-export.
    #[test]
    fn reproduces_ruby_sort_by_tie_order() {
        let fixtures: &[(Vec<i64>, Vec<usize>)] = &[
            (vec![3, 3, 3], vec![0, 1, 2]),
            (vec![2, 3, 1, 3, 3, 2], vec![2, 0, 5, 1, 3, 4]),
            (
                vec![2, 1, 2, 3, 0, 1, 0, 1, 2, 1, 2, 1, 2, 1, 1, 1, 3, 0, 3, 3, 1, 2, 0, 0, 1],
                vec![
                    23, 22, 4, 6, 17, 24, 1, 5, 7, 9, 20, 11, 13, 14, 15, 21, 12, 10, 8, 2, 0,
                    16, 18, 19, 3,
                ],
            ),
            (
                vec![
                    3, 2, 1, 0, 0, 0, 1, 3, 1, 3, 1, 2, 2, 3, 2, 0, 0, 2, 2, 2, 0, 0, 0, 1, 0,
                    1, 3, 2, 2, 3, 2, 0, 3, 0, 1, 2, 2, 2, 3, 3, 3, 0, 1, 3, 0, 3, 2, 3, 0,
                ],
                vec![
                    48, 44, 41, 33, 31, 24, 22, 21, 20, 16, 15, 4, 5, 3, 42, 2, 10, 25, 34, 8,
                    6, 23, 28, 30, 35, 36, 37, 1, 46, 27, 11, 19, 18, 17, 12, 14, 0, 47, 45,
                    43, 13, 40, 39, 38, 9, 7, 32, 26, 29,
                ],
            ),
        ];
        for (keys, expected) in fixtures {
            let mut items: Vec<usize> = (0..keys.len()).collect();
            sort_by(&mut items, |index| keys[*index]);
            assert_eq!(&items, expected);
        }
    }

    #[test]
    fn sorts_by_key_regardless_of_ties() {
        let mut items = vec![5, 3, 5, 1, 3, 1];
        sort_by(&mut items, |item| *item);
        assert_eq!(items, vec![1, 1, 3, 3, 5, 5]);
    }

    #[test]
    fn float_keys_match_integer_ordering() {
        let keys = vec![3.0, 1.0, 2.0, 1.0];
        let mut items: Vec<usize> = (0..keys.len()).collect();
        sort_by(&mut items, |index| keys[*index]);
        let sorted_keys: Vec<f64> = items.iter().map(|i| keys[*i]).collect();
        assert_eq!(sorted_keys, vec![1.0, 1.0, 2.0, 3.0]);
    }
}
