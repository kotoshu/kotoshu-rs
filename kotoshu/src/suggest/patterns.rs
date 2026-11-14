//! Shared single-edit typo-pattern predicates (the corpus's dominant
//! error classes), mirroring the gem's
//! `Kotoshu::Suggestions::Patterns`.
//!
//! Used by the composite band merge to prioritize candidates
//! regardless of which strategy produced them, by the SymSpell slate
//! sort as a channel prior, and by EditDistance's scoring.

/// True when `suggestion` fixes `original` by removing or adding
/// exactly one doubled letter ("helo"->"hello", "ommitted"->"omitted").
pub fn double_letter_fix(original: &str, suggestion: &str) -> bool {
    let os: Vec<char> = original.chars().collect();
    let ss: Vec<char> = suggestion.chars().collect();
    if (os.len() as isize - ss.len() as isize).abs() != 1 {
        return false;
    }
    let (longer, shorter) = if os.len() > ss.len() { (&os, &ss) } else { (&ss, &os) };
    for i in 0..longer.len().saturating_sub(1) {
        if longer[i] != longer[i + 1] {
            continue;
        }
        // drop ONE of the doubled pair (index i+1): the longer side
        // with the duplicate removed must equal the shorter side
        let mut reconstructed: Vec<char> = longer[..=i].to_vec();
        reconstructed.extend_from_slice(&longer[i + 2..]);
        if reconstructed == *shorter {
            return true;
        }
    }
    false
}

/// Channel prior for a single-edit pair: how typical is this typo of
/// `suggestion`? Lower = more common error class.
///   1 transposition (the most common typing error)
///   2 length-changing edit (missing / extra keystroke)
///   3 substitution of one letter
///   4 anything else (two non-cross substitutions, distance > 2)
pub fn single_edit_op(original: &str, suggestion: &str) -> u8 {
    let os: Vec<char> = original.chars().collect();
    let ss: Vec<char> = suggestion.chars().collect();
    if (os.len() as isize - ss.len() as isize).abs() > 2 {
        return 4;
    }
    match os.len().abs_diff(ss.len()) {
        0 => {
            let diff: Vec<usize> = (0..os.len()).filter(|&i| os[i] != ss[i]).collect();
            match diff.len() {
                0 => 1,
                1 => 3,
                2 => {
                    let (i, j) = (diff[0], diff[1]);
                    if os[i] == ss[j] && os[j] == ss[i] {
                        1
                    } else {
                        4
                    }
                }
                _ => 4,
            }
        }
        1 => 2,
        2 => {
            if subsequence_exactly_two(&os, &ss) {
                // both ops are the same kind of keystroke miss (two
                // dropped or two doubled letters:
                // "containg"->"containing")
                2
            } else {
                4
            }
        }
        _ => 4,
    }
}

/// True when the longer char array reduces to the shorter by deleting
/// exactly two characters (greedy subsequence match; the length guard
/// makes success mean exactly two skips).
fn subsequence_exactly_two(longer: &[char], shorter: &[char]) -> bool {
    let (longer, shorter) = if longer.len() >= shorter.len() {
        (longer, shorter)
    } else {
        (shorter, longer)
    };
    if longer.len() - shorter.len() != 2 {
        return false;
    }
    let mut li = 0usize;
    for &c in shorter {
        while li < longer.len() && longer[li] != c {
            li += 1;
        }
        if li >= longer.len() {
            return false;
        }
        li += 1;
    }
    true
}

/// Ruby `String#casecmp?` — Unicode case-FOLD equality (Onigmo full
/// fold, not plain lowercasing). ASCII pairs short-circuit through
/// `eq_ignore_ascii_case` (identical semantics); otherwise both sides
/// fold — sharp s (ß/ẞ) expands to "ss", final sigma (ς) folds to σ,
/// everything else lowercases — before the compare. The composite's
/// typed-word reject runs through this: the gem's
/// `merged.reject { |s| s.word.casecmp?(typed) }` drops a MAP-expanded
/// form ("groß") as fold-equal to the typed "gross", so the MAP
/// fixtures see zero suggestions; an ASCII-only compare would keep it.
pub(crate) fn casecmp_eq(a: &str, b: &str) -> bool {
    if a.eq_ignore_ascii_case(b) {
        return true;
    }
    fold_case(a) == fold_case(b)
}

fn fold_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            'ß' | 'ẞ' => out.push_str("ss"),
            'ς' => out.push('σ'),
            other => out.extend(other.to_lowercase()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_letter_fix_matches_ruby() {
        assert!(double_letter_fix("ommitted", "omitted"));
        assert!(double_letter_fix("helo", "hello"));
        assert!(!double_letter_fix("specifiy", "specify"));
    }

    #[test]
    fn single_edit_op_matches_ruby() {
        assert_eq!(single_edit_op("specifiy", "specify"), 2);
        assert_eq!(single_edit_op("specifiy", "specific"), 3);
        assert_eq!(single_edit_op("colums", "columns"), 2);
        assert_eq!(single_edit_op("colums", "column"), 3);
        assert_eq!(single_edit_op("teh", "the"), 1);
        assert_eq!(single_edit_op("containg", "containing"), 2);
    }

    #[test]
    fn casecmp_eq_matches_ruby_full_case_fold() {
        // MRI: "groß".casecmp?("gross") => true, "größe".casecmp?("grösse")
        // => true — the fold expands ß to "ss" instead of comparing chars.
        assert!(casecmp_eq("gross", "GROSS"));
        assert!(casecmp_eq("groß", "gross"));
        assert!(casecmp_eq("GRÖSSE", "größe"));
        assert!(!casecmp_eq("grüße", "grusse"));
        assert!(!casecmp_eq("helo", "hello"));
    }
}

#[cfg(test)]
mod probe_tests {
    use super::*;
    #[test]
    fn probe_aab() {
        println!("PROBE double_letter_fix(aaab..) aab->ab = {}", double_letter_fix("aab", "ab"));
        println!("PROBE single_edit_op(aab, ab) = {}", single_edit_op("aab", "ab"));
        println!("PROBE single_edit_op(aab, arab) = {}", single_edit_op("aab", "arab"));
    }
}
