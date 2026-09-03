//! Byte-exact port of macOS Libc `qsort_r` (Libc-1439.141.1,
//! `stdlib/FreeBSD/qsort.c` + `heapsort.c`).
//!
//! MRI's `Array#sort!` delegates to `ruby_qsort`, and on macOS builds
//! (`HAVE_BSD_QSORT_R`) that is the platform libc's `qsort_r` — a
//! FreeBSD-style fat-partition quicksort with insertion sort for small
//! spans, a swap-count insertion-sort fallback, and heapsort at recursion
//! depth exhaustion. Its tie order is deterministic but differs from a
//! stable sort and from the qs6 fallback Ruby uses on other platforms.
//!
//! The gem's `SuggestionSet#sort!` ranks suggestions through exactly this
//! sort (the conformance vectors were exported on macOS/arm64 Ruby 3.4.8
//! linking Apple's libc), and 16 of the frozen vectors contain
//! full-comparator-tie suggestion pairs whose surviving `source` label is
//! decided by this tie order. Reproducing the vectors byte-for-byte
//! requires this port, not just any total sort. On other platforms a Ruby
//! process would emit a different (equally arbitrary) tie order; the
//! vectors freeze the export platform.

/// Sort `items` with `compare`, reproducing macOS Libc `qsort_r` including
/// its tie order. `T: Clone` covers the heapsort fallback's displaced
/// element (the C code memmoves it aside).
pub fn sort_by<T: Clone>(items: &mut [T], compare: impl Fn(&T, &T) -> std::cmp::Ordering + Copy) {
    let n = items.len();
    if n <= 1 {
        return;
    }
    // DEPTH(n) = 2 * (fls(n) - 1) and fls(n) = ilog2(n) + 1 for n >= 1.
    qsort(items, 0, n, 2 * n.ilog2() as isize, compare);
}

/// `_isort` — plain insertion sort; returns false when the swap limit
/// trips (`swap_limit` 0 = unlimited, as in the small-span call).
fn isort<T>(
    items: &mut [T],
    lo: usize,
    n: usize,
    compare: &impl Fn(&T, &T) -> std::cmp::Ordering,
    swap_limit: usize,
) -> bool {
    let mut swap_cnt: usize = 0;
    for pm in lo + 1..lo + n {
        let mut pl = pm;
        while pl > lo && compare(&items[pl - 1], &items[pl]) == std::cmp::Ordering::Greater {
            items.swap(pl, pl - 1);
            if swap_limit != 0 {
                swap_cnt += 1;
                if swap_cnt > swap_limit {
                    return false;
                }
            }
            pl -= 1;
        }
    }
    true
}

/// `med3` — median of three positions.
fn med3<T>(
    items: &[T],
    a: usize,
    b: usize,
    c: usize,
    compare: &impl Fn(&T, &T) -> std::cmp::Ordering,
) -> usize {
    use std::cmp::Ordering::*;
    if compare(&items[a], &items[b]) == Less {
        if compare(&items[b], &items[c]) == Less {
            b
        } else if compare(&items[a], &items[c]) == Less {
            c
        } else {
            a
        }
    } else if compare(&items[b], &items[c]) == Greater {
        b
    } else if compare(&items[a], &items[c]) == Less {
        a
    } else {
        c
    }
}

/// `_qsort` — fat-partition quicksort over `items[lo..lo + n]`, iterating
/// on the larger side.
fn qsort<T: Clone>(
    items: &mut [T],
    lo: usize,
    n: usize,
    mut depth_limit: isize,
    compare: impl Fn(&T, &T) -> std::cmp::Ordering + Copy,
) {
    use std::cmp::Ordering::*;
    let mut lo = lo;
    let mut n = n;
    loop {
        // C: `if (depth_limit-- <= 0) { __heapsort_r(...); return; }`
        if depth_limit <= 0 {
            heapsort(items, lo, n, &compare);
            return;
        }
        depth_limit -= 1;

        if n <= 7 {
            isort(items, lo, n, &compare, 0);
            return;
        }

        // Pseudomedian: 3 samples, 9 for spans above 40.
        let pl = lo;
        let pm = lo + n / 2;
        let pn = lo + n - 1;
        let (pl, pm, pn) = if n > 40 {
            let d = n / 8;
            (
                med3(items, pl, pl + d, pl + 2 * d, &compare),
                med3(items, pm - d, pm, pm + d, &compare),
                med3(items, pn - 2 * d, pn - d, pn, &compare),
            )
        } else {
            (pl, pm, pn)
        };
        let pm = med3(items, pl, pm, pn, &compare);

        // Pull the median to the front, then fat-partition around it.
        items.swap(lo, pm);
        let mut pa = lo + 1;
        let mut pb = lo + 1;
        let mut pc = lo + n - 1;
        let mut pd = lo + n - 1;
        let mut swap_cnt = false;
        loop {
            while pb <= pc {
                match compare(&items[pb], &items[lo]) {
                    Greater => break,
                    Equal => {
                        swap_cnt = true;
                        items.swap(pa, pb);
                        pa += 1;
                        pb += 1;
                    }
                    Less => pb += 1,
                }
            }
            while pb <= pc {
                match compare(&items[pc], &items[lo]) {
                    Less => break,
                    Equal => {
                        swap_cnt = true;
                        items.swap(pc, pd);
                        pd -= 1;
                        pc -= 1;
                    }
                    Greater => pc -= 1,
                }
            }
            if pb > pc {
                break;
            }
            items.swap(pb, pc);
            swap_cnt = true;
            pb += 1;
            pc -= 1;
        }

        // Swap the `== pivot` regions into the middle.
        let end = lo + n;
        let d1 = (pa - lo).min(pb - pa);
        vecswap(items, lo, pb - d1, d1);
        let d1 = (pd - pc).min(end - pd - 1);
        vecswap(items, pb, end - d1, d1);

        if !swap_cnt {
            // Nearly-sorted input: insertion sort with a swap budget of
            // 1 + n/4; on failure fall through to partitioning.
            let r = 1 + n / 4;
            if isort(items, lo, n, &compare, r) {
                return;
            }
        }

        let d1 = pb - pa; // element count of the `< pivot` region
        let d2 = pd - pc; // element count of the `> pivot` region
        if d1 <= d2 {
            if d1 > 1 {
                qsort(items, lo, d1, depth_limit, compare);
            }
            if d2 > 1 {
                lo = end - d2;
                n = d2;
                continue; // iterate on the right partition
            }
        } else {
            if d2 > 1 {
                qsort(items, end - d2, d2, depth_limit, compare);
            }
            if d1 > 1 {
                n = d1;
                continue; // iterate on the left partition
            }
        }
        return;
    }
}

fn vecswap<T>(items: &mut [T], mut a: usize, mut b: usize, mut count: usize) {
    while count > 0 {
        items.swap(a, b);
        a += 1;
        b += 1;
        count -= 1;
    }
}

/// Apple Libc `heapsort` (1-indexed heap, displaced-element `SELECT`).
/// Reached only at quicksort depth exhaustion — never for the engine's
/// bounded suggestion pools, but ported for fidelity.
fn heapsort<T: Clone>(
    items: &mut [T],
    lo: usize,
    n: usize,
    compare: &impl Fn(&T, &T) -> std::cmp::Ordering,
) {
    use std::cmp::Ordering::*;
    if n <= 1 {
        return;
    }
    // 1-indexed: heap element i lives at items[lo + i - 1].
    let at = |i: usize| lo + i - 1;

    // CREATE: build the max-heap, sifting each parent down with swaps.
    let mut l = n / 2 + 1;
    while l > 1 {
        l -= 1;
        let mut par_i = l;
        loop {
            let mut child_i = par_i * 2;
            if child_i > n {
                break;
            }
            if child_i < n && compare(&items[at(child_i)], &items[at(child_i + 1)]) == Less {
                child_i += 1;
            }
            if compare(&items[at(child_i)], &items[at(par_i)]) != Greater {
                break;
            }
            items.swap(at(par_i), at(child_i));
            par_i = child_i;
        }
    }

    // Extraction: save the max (root) into its final slot, displace the
    // last element, then SELECT it into place along the root-to-leaf path.
    let mut nmemb = n;
    while nmemb > 1 {
        let displaced = items[at(nmemb)].clone();
        items[at(nmemb)] = items[at(1)].clone();
        nmemb -= 1;

        // SELECT phase 1: hole descends from the root, always copying the
        // larger child up (no comparison against the displaced element).
        let mut hole = 1usize;
        loop {
            let mut child_i = hole * 2;
            if child_i > nmemb {
                break;
            }
            if child_i < nmemb && compare(&items[at(child_i)], &items[at(child_i + 1)]) == Less {
                child_i += 1;
            }
            items[at(hole)] = items[at(child_i)].clone();
            hole = child_i;
        }
        // SELECT phase 2: walk the hole up while the displaced element is
        // >= its parent; then place it.
        loop {
            if hole == 1 {
                items[at(1)] = displaced;
                break;
            }
            let parent = hole / 2;
            if compare(&displaced, &items[at(parent)]) == Less {
                items[at(hole)] = displaced;
                break;
            }
            items[at(hole)] = items[at(parent)].clone();
            hole = parent;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tie permutations frozen from Ruby 3.4.8 on macOS/arm64
    /// (`Array#sort!`), which routes through Apple Libc `qsort_r`.
    #[test]
    fn reproduces_macos_qsort_tie_order() {
        let fixtures: &[(Vec<i64>, Vec<usize>)] = &[
            (vec![3, 3, 3], vec![0, 1, 2]),
            (vec![2, 3, 1, 3, 3, 2], vec![2, 0, 5, 1, 3, 4]),
            (
                vec![
                    1, 2, 3, 3, 0, 2, 3, 2, 0, 1, 3, 1, 0, 0, 0, 1, 1, 1, 3, 1, 3, 1, 0, 1, 0, 1,
                    0, 1, 0, 0, 0, 2, 0, 2, 0, 2, 3, 3, 1, 2, 1, 2, 2, 1, 1, 2, 3, 0, 3, 1, 2, 3,
                    2, 0, 2, 3, 3, 2, 2, 0,
                ],
                vec![
                    47, 34, 28, 32, 26, 30, 22, 8, 24, 59, 29, 53, 12, 13, 14, 4, 0, 9, 11, 15, 16,
                    17, 19, 21, 23, 25, 27, 38, 40, 43, 44, 49, 5, 39, 1, 50, 33, 52, 41, 42, 54,
                    57, 58, 7, 31, 45, 35, 46, 36, 37, 55, 56, 6, 3, 48, 51, 2, 20, 18, 10,
                ],
            ),
            (
                vec![
                    3, 2, 1, 0, 0, 0, 1, 3, 1, 3, 1, 2, 2, 3, 2, 0, 0, 2, 2, 2, 0, 0, 0, 1, 0, 1,
                    3, 2, 2, 3, 2, 0, 3, 0, 1, 2, 2, 2, 3, 3, 3, 0, 1, 3, 0, 3, 2, 3, 0,
                ],
                vec![
                    22, 3, 4, 15, 16, 5, 48, 20, 21, 24, 41, 44, 33, 31, 6, 42, 25, 10, 8, 2, 34,
                    23, 30, 1, 11, 12, 14, 17, 18, 19, 27, 28, 35, 36, 37, 46, 45, 47, 38, 39, 40,
                    26, 13, 43, 9, 32, 0, 29, 7,
                ],
            ),
        ];
        for (keys, expected) in fixtures {
            let mut items: Vec<usize> = (0..keys.len()).collect();
            sort_by(&mut items, |a, b| keys[*a].cmp(&keys[*b]));
            assert_eq!(&items, expected);
        }
    }

    #[test]
    fn sorts_correctly_regardless_of_ties() {
        let mut items = vec![5, 3, 5, 1, 3, 1];
        sort_by(&mut items, |a, b| a.cmp(b));
        assert_eq!(items, vec![1, 1, 3, 3, 5, 5]);
    }
}
