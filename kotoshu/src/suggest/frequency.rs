//! Kelly Project frequency tiers feeding the edit-distance strategy's
//! ranking bonus (the gem's `EditDistanceStrategy#frequency_bonus` via
//! `FrequencyProvider`).

// The generated tier tables live beside this file (generated data, kept
// out of hand-written source).
#[path = "frequency_data.rs"]
mod data;

/// The bonus for a word: 200 in the top 50, 100 in the top 200, 50 in the
/// top 1000, else 0. Tiers are cumulative, so the membership order below
/// mirrors the gem's precedence checks.
pub fn bonus(word: &str) -> u32 {
    let downcased = word.to_lowercase();
    if contains(data::TOP_50, &downcased) {
        200
    } else if contains(data::TOP_200, &downcased) {
        100
    } else if contains(data::TOP_1000, &downcased) {
        50
    } else {
        0
    }
}

fn contains(tier: &[&str], word: &str) -> bool {
    tier.binary_search(&word).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiers_carry_the_frozen_overlap_words() {
        assert_eq!(bonus("the"), 200);
        assert_eq!(bonus("help"), 100);
        assert_eq!(bonus("hello"), 50);
        assert_eq!(bonus("computer"), 50);
        assert_eq!(bonus("zzzznotaword"), 0);
    }

    #[test]
    fn capitalized_tier_entries_never_match() {
        // Kelly ships "London" capitalized; the gem matches the downcased
        // suggestion against the exact entry, so it can never hit.
        assert_eq!(bonus("london"), 0);
    }

    #[test]
    fn tiers_are_sorted_for_binary_search() {
        fn sorted(tier: &[&str]) -> bool {
            tier.windows(2).all(|w| w[0] < w[1])
        }
        assert!(sorted(data::TOP_50));
        assert!(sorted(data::TOP_200));
        assert!(sorted(data::TOP_1000));
    }
}
