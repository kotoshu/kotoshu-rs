//! Edit-distance algorithms, ported from the gem's
//! `Kotoshu::Algorithms::EditDistance` (Damerau-Levenshtein with an
//! early-exit threshold) and the plain Levenshtein used internally by the
//! phonetic and keyboard-proximity strategies. Character-based throughout
//! — Ruby `String#length` / `[]` are character operations.

/// Damerau-Levenshtein distance with early-exit threshold (the gem's
/// `distance_with_threshold`).
///
/// Returns `None` when the true distance exceeds `threshold` (row-minimum
/// early termination plus the length pre-filter). Transpositions cost 1.
pub fn damerau_with_threshold(str1: &[char], str2: &[char], threshold: usize) -> Option<usize> {
    if str1 == str2 {
        return Some(0);
    }
    if str1.is_empty() {
        return Some(str2.len());
    }
    if str2.is_empty() {
        return Some(str1.len());
    }
    if str1.len().abs_diff(str2.len()) > threshold {
        return None;
    }

    let len1 = str1.len();
    let len2 = str2.len();
    // Full matrix: the transposition step needs d[i-2][j-2].
    let mut d = vec![vec![0usize; len2 + 1]; len1 + 1];
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in d[0].iter_mut().enumerate() {
        *cell = j;
    }

    for i in 1..=len1 {
        let mut row_min = usize::MAX;
        for j in 1..=len2 {
            let cost = usize::from(str1[i - 1] != str2[j - 1]);
            let mut cell = (d[i - 1][j] + 1).min(d[i][j - 1] + 1).min(d[i - 1][j - 1] + cost);
            if i > 1 && j > 1 && str1[i - 1] == str2[j - 2] && str1[i - 2] == str2[j - 1] {
                cell = cell.min(d[i - 2][j - 2] + 1);
            }
            d[i][j] = cell;
            row_min = row_min.min(cell);
        }
        if row_min > threshold {
            return None;
        }
    }

    let result = d[len1][len2];
    (result <= threshold).then_some(result)
}

/// Plain Levenshtein distance (substitution/insertion/deletion). The gem
/// implements this three times over inside the strategies; all three
/// compute the same value.
pub fn levenshtein(str1: &[char], str2: &[char]) -> usize {
    if str1.is_empty() {
        return str2.len();
    }
    if str2.is_empty() {
        return str1.len();
    }
    let len1 = str1.len();
    let len2 = str2.len();
    let mut previous: Vec<usize> = (0..=len2).collect();
    let mut current = vec![0usize; len2 + 1];
    for i in 1..=len1 {
        current[0] = i;
        for j in 1..=len2 {
            let cost = usize::from(str1[i - 1] != str2[j - 1]);
            current[j] = (current[j - 1] + 1)
                .min(previous[j] + 1)
                .min(previous[j - 1] + cost);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[len2]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chars(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    #[test]
    fn damerau_counts_transposition_as_one() {
        assert_eq!(
            damerau_with_threshold(&chars("teh"), &chars("the"), 2),
            Some(1)
        );
    }

    #[test]
    fn damerau_bails_out_over_threshold() {
        assert_eq!(
            damerau_with_threshold(&chars("abcd"), &chars("wxyz"), 2),
            None
        );
        assert_eq!(
            damerau_with_threshold(&chars("abcd"), &chars("abcde"), 2),
            Some(1)
        );
    }

    #[test]
    fn damerau_equal_strings_short_circuit() {
        assert_eq!(damerau_with_threshold(&chars("aa"), &chars("aa"), 0), Some(0));
    }

    #[test]
    fn levenshtein_basics() {
        assert_eq!(levenshtein(&chars("kitten"), &chars("sitting")), 3);
        assert_eq!(levenshtein(&chars("abc"), &chars("abc")), 0);
        assert_eq!(levenshtein(&chars(""), &chars("abc")), 3);
    }
}
