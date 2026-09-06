//! Single-edit word permutations, ported from the gem's
//! `Algorithms::Permutations` (itself the Spylls port of Hunspell's
//! suggestmgr edits). Only the four distance-1 generators the
//! `EditDistanceStrategy` edit sweep needs are ported; enumeration
//! order is load-bearing (the sweep's candidate ties resolve in
//! emission order and the conformance vectors freeze the result).
//!
//! Character-based throughout — Ruby `String#length` / `[]` are
//! character operations.

/// `Permutations.swapchar` — permutations with adjacent characters
/// swapped. For 4- and 5-letter words also produces double swaps:
/// `ahev` → `have`, `owudl` → `would`.
pub fn swapchar(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    if chars.len() < 2 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(chars.len());
    for i in 0..chars.len() - 1 {
        let mut swapped = chars[..i].to_vec();
        swapped.push(chars[i + 1]);
        swapped.push(chars[i]);
        swapped.extend_from_slice(&chars[i + 2..]);
        out.push(swapped.into_iter().collect());
    }

    // Double swaps for short words (Ruby: word[1] + word[0] +
    // (length == 5 ? word[2] : "") + word[-1] + word[-2], then
    // word[0] + word[2] + word[1] + word[-1] + word[-2]).
    match chars.len() {
        4 => {
            let doubled: String = [chars[1], chars[0], chars[3], chars[2]].iter().collect();
            out.push(doubled);
        }
        5 => {
            let doubled: String = [chars[1], chars[0], chars[2], chars[4], chars[3]]
                .iter()
                .collect();
            out.push(doubled);
            let doubled2: String = [chars[0], chars[2], chars[1], chars[4], chars[3]]
                .iter()
                .collect();
            out.push(doubled2);
        }
        _ => {}
    }

    out
}

/// `Permutations.badchar` — permutations with each character replaced
/// by each TRY character. Positions run inner and BACKWARD
/// (`downto(0)` in the gem); a position whose character already equals
/// the TRY character is skipped.
pub fn badchar(word: &str, try_string: Option<&str>) -> Vec<String> {
    let Some(try_string) = try_string else {
        return Vec::new();
    };
    if try_string.is_empty() {
        return Vec::new();
    }

    let chars: Vec<char> = word.chars().collect();
    let mut out = Vec::new();
    for c in try_string.chars() {
        for i in (0..chars.len()).rev() {
            if chars[i] == c {
                continue;
            }
            let mut replaced = chars[..i].to_vec();
            replaced.push(c);
            replaced.extend_from_slice(&chars[i + 1..]);
            out.push(replaced.into_iter().collect());
        }
    }
    out
}

/// `Permutations.forgotchar` — permutations with one TRY character
/// inserted at every position.
pub fn forgotchar(word: &str, try_string: Option<&str>) -> Vec<String> {
    let Some(try_string) = try_string else {
        return Vec::new();
    };
    if try_string.is_empty() {
        return Vec::new();
    }

    let chars: Vec<char> = word.chars().collect();
    let mut out = Vec::new();
    for c in try_string.chars() {
        for i in 0..=chars.len() {
            let mut inserted = chars[..i].to_vec();
            inserted.push(c);
            inserted.extend_from_slice(&chars[i..]);
            out.push(inserted.into_iter().collect());
        }
    }
    out
}

/// `Permutations.extrachar` — permutations with one character removed
/// at every position.
pub fn extrachar(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    let mut out = Vec::with_capacity(chars.len());
    for i in 0..chars.len() {
        let mut deleted = chars[..i].to_vec();
        deleted.extend_from_slice(&chars[i + 1..]);
        out.push(deleted.into_iter().collect());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(mut v: Vec<String>) -> Vec<String> {
        v.sort();
        v
    }

    #[test]
    fn swapchar_adjacent_swaps() {
        assert_eq!(swapchar("teh"), vec!["eth".to_string(), "the".to_string()]);
        // 4-letter words gain the double swap of first and last pairs.
        assert_eq!(
            sorted(swapchar("ahev")),
            sorted(vec![
                "haev".to_string(),
                "aehv".to_string(),
                "ahve".to_string(),
                "have".to_string()
            ])
        );
        // 5-letter words gain two double swaps: owudl -> would.
        assert!(swapchar("owudl").contains(&"would".to_string()));
        assert!(swapchar("a").is_empty());
        assert!(swapchar("").is_empty());
    }

    #[test]
    fn badchar_backward_positions() {
        // Inner loop is downto: for "bit" with TRY "ta" the gem emits
        // btt (t at 1), tit, then bia, bat, ait.
        assert_eq!(
            badchar("bit", Some("ta")),
            vec!["btt", "tit", "bia", "bat", "ait"]
        );
        assert!(badchar("bit", None).is_empty());
        assert!(badchar("bit", Some("")).is_empty());
    }

    #[test]
    fn forgotchar_inserts_everywhere() {
        assert_eq!(forgotchar("te", Some("a")), vec!["ate", "tae", "tea"]);
        assert!(forgotchar("te", None).is_empty());
    }

    #[test]
    fn extrachar_deletes_every_position() {
        assert_eq!(extrachar("the"), vec!["he", "te", "th"]);
        assert!(extrachar("").is_empty());
    }
}
