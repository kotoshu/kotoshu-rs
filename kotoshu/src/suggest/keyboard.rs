//! Keyboard models used by the suggestion strategies — two of them,
//! because the gem uses two:
//!
//! * [`Layout`] — the OOP `Keyboard::Layouts::QWERTY` grid
//!   (`[row, col]` positions, Manhattan distance, adjacency within 1),
//!   consumed by `EditDistanceStrategy#keyboard_penalty`.
//! * [`PROXIMITY_NEIGHBORS`] — `KeyboardProximityStrategy::KEYBOARD_LAYOUT`,
//!   a hand-written adjacency table (with the pseudo-keys `tab`, `caps`,
//!   `shift`, `enter` as multi-character "neighbors"), consumed by variant
//!   generation.
//!
//! Both are frozen as the gem defines them, quirks included.

/// QWERTY key grid (`Keyboard::Layouts::QWERTY::KEY_POSITIONS`), in
/// declaration order (adjacency iteration order is load-bearing — it
/// determines variant generation order).
const QWERTY_POSITIONS: &[(char, (u8, u8))] = &[
    // Number row
    ('`', (0, 0)),
    ('1', (0, 1)),
    ('2', (0, 2)),
    ('3', (0, 3)),
    ('4', (0, 4)),
    ('5', (0, 5)),
    ('6', (0, 6)),
    ('7', (0, 7)),
    ('8', (0, 8)),
    ('9', (0, 9)),
    ('0', (0, 10)),
    ('-', (0, 11)),
    ('=', (0, 12)),
    // Top row
    ('q', (1, 0)),
    ('w', (1, 1)),
    ('e', (1, 2)),
    ('r', (1, 3)),
    ('t', (1, 4)),
    ('y', (1, 5)),
    ('u', (1, 6)),
    ('i', (1, 7)),
    ('o', (1, 8)),
    ('p', (1, 9)),
    ('[', (1, 10)),
    (']', (1, 11)),
    ('\\', (1, 12)),
    // Home row
    ('a', (2, 0)),
    ('s', (2, 1)),
    ('d', (2, 2)),
    ('f', (2, 3)),
    ('g', (2, 4)),
    ('h', (2, 5)),
    ('j', (2, 6)),
    ('k', (2, 7)),
    ('l', (2, 8)),
    (';', (2, 9)),
    ('\'', (2, 10)),
    // Bottom row
    ('z', (3, 0)),
    ('x', (3, 1)),
    ('c', (3, 2)),
    ('v', (3, 3)),
    ('b', (3, 4)),
    ('n', (3, 5)),
    ('m', (3, 6)),
    (',', (3, 7)),
    ('.', (3, 8)),
    ('/', (3, 9)),
];

/// The QWERTY layout (`Keyboard::Registry.layout_for("en")`).
pub struct Layout {
    positions: std::collections::HashMap<char, (u8, u8)>,
}

impl Layout {
    /// Build the QWERTY layout.
    pub fn qwerty() -> Self {
        Self {
            positions: QWERTY_POSITIONS.iter().copied().collect(),
        }
    }

    /// `Layout#position` — key looked up downcased.
    fn position(&self, key: char) -> Option<(u8, u8)> {
        let lowered: Vec<char> = key.to_lowercase().collect();
        match lowered[..] {
            [c] => self.positions.get(&c).copied(),
            _ => None, // Ruby: multi-char String#downcase never hits the table
        }
    }

    /// `Layout#distance` — Manhattan distance; `None` is Ruby's
    /// `Float::INFINITY` (either key unknown).
    pub fn distance(&self, key1: char, key2: char) -> Option<u32> {
        let pos1 = self.position(key1)?;
        let pos2 = self.position(key2)?;
        Some((pos1.0.abs_diff(pos2.0) + pos1.1.abs_diff(pos2.1)) as u32)
    }
}

/// `KeyboardProximityStrategy::KEYBOARD_LAYOUT` — neighbors for variant
/// generation, pseudo-keys included (they splice multi-character strings
/// into variants, faithfully to the gem).
pub fn proximity_neighbors(key: char) -> &'static [&'static str] {
    let lowered: Vec<char> = key.to_lowercase().collect();
    let [c] = lowered[..] else {
        return &[];
    };
    match c {
        '`' => &["1", "tab"],
        '1' => &["`", "2", "q"],
        '2' => &["1", "3", "w", "q"],
        '3' => &["2", "4", "e", "w"],
        '4' => &["3", "5", "r", "e"],
        '5' => &["4", "6", "t", "r"],
        '6' => &["5", "7", "y", "t"],
        '7' => &["6", "8", "u", "y"],
        '8' => &["7", "9", "i", "u"],
        '9' => &["8", "0", "o", "i"],
        '0' => &["9", "p", "o"],
        '-' => &["0", "="],
        '=' => &["-"],
        'q' => &["tab", "w", "a", "1"],
        'w' => &["q", "e", "a", "s", "2"],
        'e' => &["w", "r", "s", "d", "3"],
        'r' => &["e", "t", "d", "f", "4"],
        't' => &["r", "y", "f", "g", "5"],
        'y' => &["t", "u", "g", "h", "6"],
        'u' => &["y", "i", "h", "j", "7"],
        'i' => &["u", "o", "j", "k", "8"],
        'o' => &["i", "p", "k", "l", "9"],
        'p' => &["o", "l", ";", "0"],
        '[' => &["p", "'"],
        ']' => &["enter", "\\"],
        '\\' => &["enter"],
        'a' => &["caps", "s", "z", "q"],
        's' => &["a", "d", "z", "x", "w"],
        'd' => &["s", "f", "x", "c", "e"],
        'f' => &["d", "g", "c", "v", "r"],
        'g' => &["f", "h", "v", "b", "t"],
        'h' => &["g", "j", "b", "n", "y"],
        'j' => &["h", "k", "n", "m", "u"],
        'k' => &["j", "l", "m", ",", "i"],
        'l' => &["k", ";", ",", ".", "o"],
        ';' => &["l", "'", ".", "p"],
        '\'' => &[";"],
        'z' => &["shift", "s", "x", "a"],
        'x' => &["z", "c", "s", "d"],
        'c' => &["x", "v", "d", "f"],
        'v' => &["c", "b", "f", "g"],
        'b' => &["v", "n", "g", "h"],
        'n' => &["b", "m", "h", "j"],
        'm' => &["n", ",", "j", "k"],
        ',' => &["m", ".", "k", "l"],
        '.' => &[",", "/", "l", ";"],
        '/' => &[".", "shift"],
        ' ' => &[],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwerty_distances() {
        let layout = Layout::qwerty();
        assert_eq!(layout.distance('q', 'w'), Some(1));
        assert_eq!(layout.distance('q', 'p'), Some(9));
        assert_eq!(layout.distance('a', 'A'), Some(0)); // same key, case-folded
        assert_eq!(layout.distance('q', 'é'), None); // not on the layout
        assert_eq!(layout.distance('q', 'q'), Some(0));
    }

    #[test]
    fn proximity_table_includes_pseudo_keys() {
        assert_eq!(proximity_neighbors('q'), &["tab", "w", "a", "1"]);
        assert_eq!(proximity_neighbors('Q'), &["tab", "w", "a", "1"]);
        assert!(proximity_neighbors('é').is_empty());
    }
}
