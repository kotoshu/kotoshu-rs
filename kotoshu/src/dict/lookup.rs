//! `correct?` lookup, ported from the gem's `Algorithms::Lookup`
//! (Spylls-derived).
//!
//! A word is correct when at least one *form* of it is valid: some
//! capitalization variant, split into stem (+ up to two prefixes and two
//! suffixes per the `.aff` rules, all mutually compatible), or as a
//! compound of valid parts. The structure mirrors the Ruby module method
//! for method: [`Lookuper::call`] → `good_forms` → `affix_forms` /
//! `compound_forms` → `produce_affix_forms` → `desuffix`/`deprefix`, with
//! `is_good_form` / `is_bad_compound` as the validity predicates.

use std::collections::BTreeSet;
use std::path::Path;

use super::LoadError;
use super::aff::{Aff, BreakPattern, parse_lines as parse_aff_lines};
use super::casing::CapType;
use super::dic::Dic;
use super::encoding;

/// The word-correctness engine over one parsed aff+dic pair.
#[derive(Debug)]
pub struct Lookuper {
    /// Parsed `.aff` data.
    pub(super) aff: Aff,
    /// Indexed `.dic` entries.
    pub(super) dic: Dic,
}

/// Position of a word part inside a compound (the gem's `CompoundPos`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompoundPos {
    Begin,
    Middle,
    End,
}

/// A hypothesis of how a word splits into stem + affixes (the gem's
/// `AffixForm`). Affix slots hold indexes into
/// [`Aff::prefixes`]/[`Aff::suffixes`]; `in_dictionary` indexes
/// `Dic::entries`.
#[derive(Debug, Clone)]
struct Form {
    text: String,
    stem: String,
    prefix: Option<usize>,
    suffix: Option<usize>,
    prefix2: Option<usize>,
    suffix2: Option<usize>,
    in_dictionary: Option<usize>,
}

impl Form {
    fn whole(word: &str) -> Self {
        Self {
            text: word.to_owned(),
            stem: word.to_owned(),
            prefix: None,
            suffix: None,
            prefix2: None,
            suffix2: None,
            in_dictionary: None,
        }
    }

    fn has_affixes(&self) -> bool {
        self.suffix.is_some() || self.prefix.is_some()
    }

    fn is_base(&self) -> bool {
        !self.has_affixes()
    }
}

impl Lookuper {
    /// Load an `.aff`/`.dic` pair (the gem's `LookupBuilder`).
    pub fn load(aff_path: &Path, dic_path: &Path) -> Result<Self, LoadError> {
        let aff_bytes = std::fs::read(aff_path)?;
        let dic_bytes = std::fs::read(dic_path)?;
        let encoding = encoding::detect(&aff_bytes).map_err(LoadError::Aff)?;

        let aff_lines = encoding::decode_lines(&aff_bytes, encoding);
        let mut aff = parse_aff_lines(aff_lines).map_err(LoadError::Aff)?;
        let flag_format = aff.flag_format;
        let aliases = std::mem::take(&mut aff.af_aliases);

        let dic_lines = encoding::decode_lines(&dic_bytes, encoding);
        let (dic, ph_reps) =
            Dic::parse(&dic_lines, flag_format, &aliases, &aff.casing, &aff.ignore);
        aff.af_aliases = aliases;
        aff.rep.extend(ph_reps);

        Ok(Self { aff, dic })
    }

    /// The outermost correctness check (the gem's `Lookuper#call`):
    /// forbidden-word gate, `ICONV`, `IGNORE`, plain numbers, then every
    /// break-pattern splitting whose parts are all correct.
    pub fn call(&self, word: &str) -> bool {
        if let Some(forbidden) = &self.aff.forbiddenword
            && self.dic.has_flag(word, forbidden, true)
        {
            return false;
        }

        let mut word_to_check = match &self.aff.iconv {
            Some(table) => table.apply(word),
            None => word.to_owned(),
        };
        if !self.aff.ignore.is_empty() {
            word_to_check = super::aff::strip_ignore(&word_to_check, &self.aff.ignore);
        }

        if is_number(&word_to_check) {
            return true;
        }

        self.break_word(&word_to_check, 0).iter().any(|parts| {
            parts
                .iter()
                .all(|part| part.is_empty() || self.good_forms_any(part))
        })
    }

    /// Whether any good form exists for the word (the gem's `good_forms`
    /// consumed by `any?`).
    fn good_forms_any(&self, word: &str) -> bool {
        let (captype, variants) = self.aff.casing.variants(word);
        for variant in &variants {
            for form in self.affix_forms(variant, captype, true, false, None, &[], &[], &[]) {
                // Special German ß handling (CHECKSHARPS + KEEPCASE).
                if let (true, Some(keepcase)) = (self.aff.checksharps, self.aff.keepcase.as_deref())
                {
                    let stem = match form.in_dictionary {
                        Some(idx) => self.dic.entries[idx].stem.as_str(),
                        None => form.stem.as_str(),
                    };
                    if stem.contains('ß')
                        && captype == CapType::All
                        && word.contains('ß')
                        && self.form_flags(&form).contains(keepcase)
                    {
                        continue;
                    }
                }
                return true;
            }
            if self.compound_forms_any(variant, captype) {
                return true;
            }
        }
        false
    }

    /// Produce every valid affix form of the word (the gem's
    /// `affix_forms_internal`).
    #[allow(clippy::too_many_arguments)]
    fn affix_forms(
        &self,
        word: &str,
        captype: CapType,
        allow_nosuggest: bool,
        with_forbidden: bool,
        compoundpos: Option<CompoundPos>,
        prefix_flags: &[String],
        suffix_flags: &[String],
        forbidden_flags: &[String],
    ) -> Vec<Form> {
        let mut out = Vec::new();
        for form in self.produce_affix_forms(
            word,
            compoundpos,
            prefix_flags,
            suffix_flags,
            forbidden_flags,
        ) {
            let mut found = false;
            let homonyms: Vec<usize> = self.dic.homonyms(&form.stem).to_vec();

            // FORBIDDENWORD: a forbidden stem must not appear as a
            // compound part or under affixes.
            if !with_forbidden
                && let Some(forbidden) = self.aff.forbiddenword.as_deref()
                && (compoundpos.is_some() || form.has_affixes())
                && homonyms
                    .iter()
                    .any(|&idx| self.dic.entries[idx].flags.contains(forbidden))
            {
                continue;
            }

            for &idx in &homonyms {
                let mut candidate = form.clone();
                candidate.in_dictionary = Some(idx);
                if self.is_good_form(&candidate, captype, allow_nosuggest, compoundpos) {
                    found = true;
                    out.push(candidate);
                }
            }

            // FORCEUCASE: at the start of a compound with a capitalized
            // input, also try lowercased stem homonyms.
            if compoundpos == Some(CompoundPos::Begin)
                && self.aff.forceucase.is_some()
                && captype == CapType::Init
            {
                let lowered = super::casing::unicode_lowercase(&form.stem);
                for &idx in self.dic.homonyms(&lowered) {
                    let mut candidate = form.clone();
                    candidate.in_dictionary = Some(idx);
                    if self.is_good_form(&candidate, captype, allow_nosuggest, compoundpos) {
                        found = true;
                        out.push(candidate);
                    }
                }
            }

            // ALLCAPS case-insensitive fallback.
            if found
                || compoundpos.is_some()
                || captype != CapType::All
                || self.aff.casing.guess(word) != CapType::No
            {
                continue;
            }
            for &idx in self.dic.homonyms_ignorecase(&form.stem) {
                if !with_forbidden
                    && let Some(forbidden) = self.aff.forbiddenword.as_deref()
                    && form.has_affixes()
                    && self.dic.entries[idx].flags.contains(forbidden)
                {
                    continue;
                }
                let mut candidate = form.clone();
                candidate.in_dictionary = Some(idx);
                if self.is_good_form(&candidate, captype, allow_nosuggest, compoundpos) {
                    out.push(candidate);
                }
            }
        }
        out
    }

    /// Whether any valid compound form exists (the gem's
    /// `compound_forms_internal` under `any?`).
    fn compound_forms_any(&self, word: &str, captype: CapType) -> bool {
        if let Some(forbidden) = self.aff.forbiddenword.as_deref() {
            let forbidden_found = self
                .affix_forms(word, captype, true, true, None, &[], &[], &[])
                .iter()
                .any(|form| self.form_flags(form).contains(forbidden));
            if forbidden_found {
                return false;
            }
        }

        if self.aff.compoundbegin.is_some() || self.aff.compoundflag.is_some() {
            for compound in self.compounds_by_flags(word, captype, 0) {
                if !self.is_bad_compound(&compound, captype) {
                    return true;
                }
            }
        }

        if !self.aff.compoundrules.is_empty() {
            let rules: Vec<usize> = (0..self.aff.compoundrules.len()).collect();
            for compound in self.compounds_by_rules(word, &[], &rules) {
                if !self.is_bad_compound(&compound, captype) {
                    return true;
                }
            }
        }

        false
    }

    /// Generate all possible affix forms (the gem's `produce_affix_forms`).
    fn produce_affix_forms(
        &self,
        word: &str,
        compoundpos: Option<CompoundPos>,
        prefix_flags: &[String],
        suffix_flags: &[String],
        forbidden_flags: &[String],
    ) -> Vec<Form> {
        let mut out = vec![Form::whole(word)];

        let suffix_allowed = compoundpos.is_none()
            || compoundpos == Some(CompoundPos::End)
            || !suffix_flags.is_empty();
        let prefix_allowed = compoundpos.is_none()
            || compoundpos == Some(CompoundPos::Begin)
            || !prefix_flags.is_empty();

        if suffix_allowed {
            out.extend(self.desuffix(word, suffix_flags, forbidden_flags, false, false));
        }

        if prefix_allowed {
            for form in self.deprefix(word, prefix_flags, forbidden_flags, false) {
                out.push(form.clone());

                // Cross-product prefix + suffix.
                if suffix_allowed
                    && let Some(prefix_idx) = form.prefix
                    && self.aff.prefixes[prefix_idx].crossproduct
                {
                    for mut form2 in
                        self.desuffix(&form.stem, suffix_flags, forbidden_flags, false, true)
                    {
                        form2.text = form.text.clone();
                        form2.prefix = Some(prefix_idx);
                        out.push(form2);
                    }
                }
            }
        }

        out
    }

    /// Remove one (or, nested once, two) suffixes (the gem's `desuffix`).
    fn desuffix(
        &self,
        word: &str,
        required_flags: &[String],
        forbidden_flags: &[String],
        nested: bool,
        crossproduct: bool,
    ) -> Vec<Form> {
        let mut out = Vec::new();
        let mut candidates: Vec<usize> =
            self.aff.suffixes_index.get("").cloned().unwrap_or_default();
        if let Some(key) = word.chars().next_back().map(|c| c.to_string())
            && let Some(bucket) = self.aff.suffixes_index.get(&key)
        {
            candidates.extend(bucket.iter().copied());
        }

        for idx in candidates {
            let suffix = &self.aff.suffixes[idx];
            if crossproduct && !suffix.crossproduct {
                continue;
            }
            if !required_flags
                .iter()
                .all(|flag| suffix.flags.contains(flag))
            {
                continue;
            }
            if forbidden_flags
                .iter()
                .any(|flag| suffix.flags.contains(flag))
            {
                continue;
            }
            if word.ends_with(suffix.add.as_str()) {
                let base = if suffix.add.is_empty() {
                    word.to_owned()
                } else {
                    drop_last_chars(word, suffix.add.chars().count()).to_owned()
                };
                let stem = format!("{}{}", base, suffix.strip);
                if let Some(condition) = &suffix.condition
                    && !condition.matches(&stem)
                {
                    continue;
                }

                let mut form = Form::whole(word);
                form.stem = stem.clone();
                form.suffix = Some(idx);
                out.push(form);

                // One more suffix level.
                if !nested {
                    let mut required: Vec<String> = Vec::with_capacity(required_flags.len() + 1);
                    required.push(suffix.flag.clone());
                    required.extend(required_flags.iter().cloned());
                    for mut form2 in
                        self.desuffix(&stem, &required, forbidden_flags, true, crossproduct)
                    {
                        form2.text = word.to_owned();
                        form2.suffix2 = Some(idx);
                        out.push(form2);
                    }
                }
            }
        }
        out
    }

    /// Remove one (or, under `COMPLEXPREFIXES`, two) prefixes (the gem's
    /// `deprefix`).
    fn deprefix(
        &self,
        word: &str,
        required_flags: &[String],
        forbidden_flags: &[String],
        nested: bool,
    ) -> Vec<Form> {
        let mut out = Vec::new();
        let mut candidates: Vec<usize> =
            self.aff.prefixes_index.get("").cloned().unwrap_or_default();
        if let Some(key) = word.chars().next().map(|c| c.to_string())
            && let Some(bucket) = self.aff.prefixes_index.get(&key)
        {
            candidates.extend(bucket.iter().copied());
        }

        for idx in candidates {
            let prefix = &self.aff.prefixes[idx];
            if !required_flags
                .iter()
                .all(|flag| prefix.flags.contains(flag))
            {
                continue;
            }
            if forbidden_flags
                .iter()
                .any(|flag| prefix.flags.contains(flag))
            {
                continue;
            }
            if word.starts_with(prefix.add.as_str()) {
                let rest = drop_first_chars(word, prefix.add.chars().count()).to_owned();
                let stem = format!("{}{}", prefix.strip, rest);
                if let Some(condition) = &prefix.condition
                    && !condition.matches(&stem)
                {
                    continue;
                }

                let mut form = Form::whole(word);
                form.stem = stem.clone();
                form.prefix = Some(idx);
                out.push(form);

                // A second prefix level under COMPLEXPREFIXES.
                if !nested && self.aff.complexprefixes {
                    let mut required: Vec<String> = Vec::with_capacity(required_flags.len() + 1);
                    required.push(prefix.flag.clone());
                    required.extend(required_flags.iter().cloned());
                    for mut form2 in self.deprefix(&stem, &required, forbidden_flags, true) {
                        form2.text = word.to_owned();
                        form2.prefix2 = Some(idx);
                        out.push(form2);
                    }
                }
            }
        }
        out
    }

    /// Validity predicate for an affix form (the gem's `is_good_form`).
    fn is_good_form(
        &self,
        form: &Form,
        captype: CapType,
        allow_nosuggest: bool,
        compoundpos: Option<CompoundPos>,
    ) -> bool {
        let Some(dict_idx) = form.in_dictionary else {
            return false;
        };
        let root_flags = &self.dic.entries[dict_idx].flags;
        let all_flags = self.form_flags(form);

        if !allow_nosuggest
            && let Some(nosuggest) = self.aff.nosuggest.as_deref()
            && root_flags.contains(nosuggest)
        {
            return false;
        }

        if let Some(keepcase) = self.aff.keepcase.as_deref()
            && root_flags.contains(keepcase)
        {
            let stem_captype = self.aff.casing.guess(&self.dic.entries[dict_idx].stem);
            let sharps_ok = self.aff.checksharps && self.dic.entries[dict_idx].stem.contains('ß');
            if captype != stem_captype && !sharps_ok {
                return false;
            }
        }

        if let Some(needaffix) = self.aff.needaffix.as_deref() {
            if root_flags.contains(needaffix) && !form.has_affixes() {
                return false;
            }
            if form.has_affixes()
                && self
                    .all_affix_cont_flags(form)
                    .iter()
                    .all(|flags| flags.contains(needaffix))
            {
                return false;
            }
        }

        if let Some(prefix_idx) = form.prefix
            && !all_flags.contains(&self.aff.prefixes[prefix_idx].flag)
        {
            return false;
        }
        if let Some(suffix_idx) = form.suffix
            && !all_flags.contains(&self.aff.suffixes[suffix_idx].flag)
        {
            return false;
        }

        if let Some(circumfix) = self.aff.circumfix.as_deref() {
            let suffix_has = form
                .suffix
                .is_some_and(|idx| self.aff.suffixes[idx].flags.contains(circumfix));
            let prefix_has = form
                .prefix
                .is_some_and(|idx| self.aff.prefixes[idx].flags.contains(circumfix));
            if suffix_has != prefix_has {
                return false;
            }
        }

        let Some(compoundpos) = compoundpos else {
            // Outside compounds, ONLYINCOMPOUND words are rejected.
            return match self.aff.onlyincompound.as_deref() {
                Some(flag) => !all_flags.contains(flag),
                None => true,
            };
        };

        if let Some(compoundflag) = self.aff.compoundflag.as_deref()
            && all_flags.contains(compoundflag)
        {
            return true;
        }
        let position_flag = match compoundpos {
            CompoundPos::Begin => self.aff.compoundbegin.as_deref(),
            CompoundPos::Middle => self.aff.compoundmiddle.as_deref(),
            CompoundPos::End => self.aff.compoundend.as_deref(),
        };
        match position_flag {
            Some(flag) => all_flags.contains(flag),
            None => false,
        }
    }

    /// Compounds by COMPOUNDFLAG/COMPOUNDBEGIN/…(the gem's
    /// `compounds_by_flags`).
    fn compounds_by_flags(
        &self,
        word_rest: &str,
        captype: CapType,
        depth: usize,
    ) -> Vec<Vec<Form>> {
        let mut out = Vec::new();
        let compound_min = self.aff.compoundmin.unwrap_or(3);
        let permit: Vec<String> = self.aff.compoundpermitflag.clone().into_iter().collect();
        let forbid: Vec<String> = self.aff.compoundforbidflag.clone().into_iter().collect();

        // Rest as compound end.
        if depth > 0 {
            for form in self.affix_forms(
                word_rest,
                captype,
                true,
                false,
                Some(CompoundPos::End),
                &permit,
                &[],
                &forbid,
            ) {
                out.push(vec![form]);
            }
        }

        let length = word_rest.chars().count() as i64;
        if length < compound_min * 2 {
            return out;
        }
        if let Some(word_max) = self.aff.compoundwordmax
            && depth as i64 >= word_max
        {
            return out;
        }

        let compoundpos = if depth == 0 {
            CompoundPos::Begin
        } else {
            CompoundPos::Middle
        };
        let prefix_flags: Vec<String> = if compoundpos == CompoundPos::Begin {
            Vec::new()
        } else {
            permit.clone()
        };

        for pos in compound_min..(length - compound_min + 1) {
            let beg = first_chars(word_rest, pos as usize);
            let rest = drop_first_chars(word_rest, pos as usize);

            for form in self.affix_forms(
                beg,
                captype,
                true,
                false,
                Some(compoundpos),
                &prefix_flags,
                &permit,
                &forbid,
            ) {
                for partial in self.compounds_by_flags(rest, captype, depth + 1) {
                    let mut compound = Vec::with_capacity(partial.len() + 1);
                    compound.push(form.clone());
                    compound.extend(partial);
                    out.push(compound);
                }
            }

            // SIMPLIFIEDTRIPLE handling.
            if self.aff.simplifiedtriple
                && !beg.is_empty()
                && !rest.is_empty()
                && beg.chars().next_back() == rest.chars().next()
            {
                let beg_tripled: String = beg
                    .chars()
                    .chain(std::iter::once(beg.chars().next_back().unwrap()))
                    .collect();

                for form in self.affix_forms(
                    &beg_tripled,
                    captype,
                    true,
                    false,
                    Some(compoundpos),
                    &prefix_flags,
                    &permit,
                    &forbid,
                ) {
                    for partial in self.compounds_by_flags(rest, captype, depth + 1) {
                        let mut compound = Vec::with_capacity(partial.len() + 1);
                        let mut form = form.clone();
                        form.text = beg.to_owned();
                        compound.push(form);
                        compound.extend(partial);
                        out.push(compound);
                    }
                }
            }
        }
        out
    }

    /// Compounds by COMPOUNDRULE (the gem's `compounds_by_rules`).
    /// `prev_parts` are dictionary entry indexes of the parts chosen so
    /// far; `rules` the indexes of still-matching compound rules.
    fn compounds_by_rules(
        &self,
        word_rest: &str,
        prev_parts: &[usize],
        rules: &[usize],
    ) -> Vec<Vec<Form>> {
        let mut out = Vec::new();
        let compound_min = self.aff.compoundmin.unwrap_or(3);
        let compound_word_max = self.aff.compoundwordmax;

        // The rest as the final part.
        if !prev_parts.is_empty() {
            for &homonym in self.dic.homonyms(word_rest) {
                let mut parts = prev_parts.to_vec();
                parts.push(homonym);
                let flag_sets: Vec<BTreeSet<String>> = parts
                    .iter()
                    .map(|&idx| self.dic.entries[idx].flags.clone())
                    .collect();
                if rules
                    .iter()
                    .any(|&rule| self.aff.compoundrules[rule].full_match(&flag_sets))
                {
                    out.push(vec![Form::whole(word_rest)]);
                }
            }
        }

        let length = word_rest.chars().count() as i64;
        if length < compound_min * 2 {
            return out;
        }
        if let Some(word_max) = compound_word_max
            && prev_parts.len() as i64 >= word_max
        {
            return out;
        }

        for pos in compound_min..(length - compound_min + 1) {
            let beg = first_chars(word_rest, pos as usize);
            for &homonym in self.dic.homonyms(beg) {
                let mut parts = prev_parts.to_vec();
                parts.push(homonym);
                let flag_sets: Vec<BTreeSet<String>> = parts
                    .iter()
                    .map(|&idx| self.dic.entries[idx].flags.clone())
                    .collect();
                let matching: Vec<usize> = rules
                    .iter()
                    .copied()
                    .filter(|&rule| self.aff.compoundrules[rule].partial_match(&flag_sets))
                    .collect();
                if matching.is_empty() {
                    continue;
                }
                for partial in self.compounds_by_rules(
                    drop_first_chars(word_rest, pos as usize),
                    &parts,
                    &matching,
                ) {
                    let mut compound = Vec::with_capacity(partial.len() + 1);
                    compound.push(Form::whole(beg));
                    compound.extend(partial);
                    out.push(compound);
                }
            }
        }
        out
    }

    /// Compound rejection predicate (the gem's `is_bad_compound`).
    fn is_bad_compound(&self, compound: &[Form], captype: CapType) -> bool {
        // FORCEUCASE.
        if let Some(forceucase) = self.aff.forceucase.as_deref()
            && captype != CapType::All
            && captype != CapType::Init
            && let Some(last) = compound.last()
            && self.dic.has_flag(&last.text, forceucase, false)
        {
            return true;
        }

        for idx in 0..compound.len().saturating_sub(1) {
            let left = &compound[idx];
            let right = &compound[idx + 1];

            // COMPOUNDFORBIDFLAG.
            if let Some(forbid) = self.aff.compoundforbidflag.as_deref()
                && self.dic.has_flag(&left.text, forbid, false)
            {
                return true;
            }

            // "left right" existing as a single (possibly affixed) entry.
            let joined = format!("{} {}", left.text, right.text);
            if !self
                .affix_forms(&joined, captype, true, false, None, &[], &[], &[])
                .is_empty()
            {
                return true;
            }

            // CHECKCOMPOUNDREP.
            if self.aff.checkcompoundrep {
                for candidate in self.replchars(&format!("{}{}", left.text, right.text)) {
                    if !self
                        .affix_forms(&candidate, captype, true, false, None, &[], &[], &[])
                        .is_empty()
                    {
                        return true;
                    }
                }
            }

            // CHECKCOMPOUNDTRIPLE.
            if self.aff.checkcompoundtriple {
                let left_chars: Vec<char> = left.text.chars().collect();
                let right_chars: Vec<char> = right.text.chars().collect();
                if right_chars.is_empty() || left_chars.is_empty() {
                    continue;
                }
                let case_a: Vec<char> = left_chars[left_chars.len().saturating_sub(2)..]
                    .iter()
                    .copied()
                    .chain(std::iter::once(right_chars[0]))
                    .collect();
                let case_b: Vec<char> = std::iter::once(left_chars[left_chars.len() - 1])
                    .chain(right_chars.iter().take(2).copied())
                    .collect();
                let uniq = |chars: &[char]| {
                    let mut sorted = chars.to_vec();
                    sorted.sort_unstable();
                    sorted.dedup();
                    sorted.len() == 1
                };
                if uniq(&case_a) || uniq(&case_b) {
                    return true;
                }
            }

            // CHECKCOMPOUNDCASE.
            if self.aff.checkcompoundcase {
                let right_c = right.text.chars().next();
                let left_c = left.text.chars().next_back();
                if let (Some(right_c), Some(left_c)) = (right_c, left_c)
                    && (char_upper_equals_self(right_c) || char_upper_equals_self(left_c))
                    && right_c != '-'
                    && left_c != '-'
                {
                    return true;
                }
            }

            // CHECKCOMPOUNDPATTERN.
            if self.aff.checkcompoundpatterns.iter().any(|pattern| {
                pattern.matches(
                    &self.form_stem(left),
                    &self.form_flags(left),
                    left.is_base(),
                    &self.form_stem(right),
                    &self.form_flags(right),
                    right.is_base(),
                )
            }) {
                return true;
            }

            // CHECKCOMPOUNDDUP.
            if self.aff.checkcompounddup && left.text == right.text && idx == compound.len() - 2 {
                return true;
            }
        }

        false
    }

    /// The gem's `Permutations.replchars` (string candidates only — the
    /// word-split candidates are a suggest-path concern).
    fn replchars(&self, word: &str) -> Vec<String> {
        let mut out = Vec::new();
        if word.chars().count() < 2 || self.aff.rep.is_empty() {
            return out;
        }
        for entry in &self.aff.rep {
            if entry.pattern.is_empty() {
                continue;
            }
            let mut pos = 0;
            while let Some(offset) = word[pos..].find(&entry.pattern) {
                let start = pos + offset;
                let end = start + entry.pattern.len();
                let mut suggestion = String::new();
                suggestion.push_str(&word[..start]);
                suggestion.push_str(&entry.replacement.replace('_', " "));
                suggestion.push_str(&word[end..]);
                out.push(suggestion);
                pos = end;
                if pos >= word.len() {
                    break;
                }
            }
        }
        out
    }

    /// All break-pattern splittings of the text (the gem's `break_word`).
    fn break_word(&self, text: &str, depth: usize) -> Vec<Vec<String>> {
        if depth > 10 {
            return Vec::new();
        }
        let mut out = vec![vec![text.to_owned()]];

        for pattern in &self.aff.break_patterns {
            let mut pos = 0usize; // char position
            while let Some((group_start, group_end, whole_end)) =
                find_break_match(text, pattern, pos)
            {
                let start = first_chars(text, group_start).to_owned();
                let rest = drop_first_chars(text, group_end).to_owned();
                for mut breaking in self.break_word(&rest, depth + 1) {
                    let mut split = Vec::with_capacity(breaking.len() + 1);
                    split.push(start.clone());
                    split.append(&mut breaking);
                    out.push(split);
                }
                pos = whole_end;
                if pos >= text.chars().count() {
                    break;
                }
            }
        }
        out
    }

    /// Stem for pattern matching: the dictionary stem when attached, else
    /// the form's stem.
    fn form_stem(&self, form: &Form) -> String {
        match form.in_dictionary {
            Some(idx) => self.dic.entries[idx].stem.clone(),
            None => form.stem.clone(),
        }
    }

    /// Combined flags: dictionary entry + prefix/suffix continuation flags
    /// (secondary affixes contribute none — the gem's `AffixForm#flags`).
    fn form_flags(&self, form: &Form) -> BTreeSet<String> {
        let mut flags = match form.in_dictionary {
            Some(idx) => self.dic.entries[idx].flags.clone(),
            None => BTreeSet::new(),
        };
        if let Some(idx) = form.prefix {
            flags.extend(self.aff.prefixes[idx].flags.iter().cloned());
        }
        if let Some(idx) = form.suffix {
            flags.extend(self.aff.suffixes[idx].flags.iter().cloned());
        }
        flags
    }

    /// Continuation flags of every affix slot, primary and secondary (the
    /// gem's `all_affixes`).
    fn all_affix_cont_flags(&self, form: &Form) -> Vec<&BTreeSet<String>> {
        let mut out = Vec::with_capacity(4);
        if let Some(idx) = form.prefix2 {
            out.push(&self.aff.prefixes[idx].flags);
        }
        if let Some(idx) = form.prefix {
            out.push(&self.aff.prefixes[idx].flags);
        }
        if let Some(idx) = form.suffix {
            out.push(&self.aff.suffixes[idx].flags);
        }
        if let Some(idx) = form.suffix2 {
            out.push(&self.aff.suffixes[idx].flags);
        }
        out
    }
}

/// Ruby's `/^\d+(\.\d+)?$/`.
fn is_number(word: &str) -> bool {
    let Some((int_part, rest)) = word.split_once('.') else {
        return !word.is_empty() && word.chars().all(|c| c.is_ascii_digit());
    };
    !int_part.is_empty()
        && int_part.chars().all(|c| c.is_ascii_digit())
        && !rest.is_empty()
        && rest.chars().all(|c| c.is_ascii_digit())
}

fn char_upper_equals_self(c: char) -> bool {
    let mut upper = c.to_uppercase();
    upper.next() == Some(c) && upper.next().is_none()
}

fn first_chars(text: &str, n: usize) -> &str {
    text.char_indices()
        .nth(n)
        .map_or(text, |(idx, _)| &text[..idx])
}

fn drop_first_chars(text: &str, n: usize) -> &str {
    let mut remaining = n;
    for (idx, _) in text.char_indices() {
        if remaining == 0 {
            return &text[idx..];
        }
        remaining -= 1;
    }
    ""
}

fn drop_last_chars(text: &str, n: usize) -> &str {
    let count = text.chars().count();
    first_chars(text, count.saturating_sub(n))
}

/// Find the next break-pattern match at or after char position `from`:
/// returns (group start, group end, whole-match end), all char positions
/// (the gem matches `.(pattern).` for unanchored patterns).
fn find_break_match(
    text: &str,
    pattern: &BreakPattern,
    from: usize,
) -> Option<(usize, usize, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let plen = pattern.pattern.chars().count();
    if plen == 0 {
        return None;
    }
    let matches_at = |start: usize| -> bool {
        chars[start..start + plen]
            .iter()
            .zip(pattern.pattern.chars())
            .all(|(a, b)| *a == b)
    };

    if pattern.anchor_start && pattern.anchor_end {
        // "(^pat$)" — the whole text, only at position 0.
        if from == 0 && chars.len() == plen && matches_at(0) {
            return Some((0, plen, plen));
        }
        return None;
    }
    if pattern.anchor_start {
        // The match must start at 0; the search position must not have
        // passed it.
        if from == 0 && plen <= chars.len() && matches_at(0) {
            return Some((0, plen, plen));
        }
        return None;
    }
    if pattern.anchor_end {
        let start = chars.len().checked_sub(plen)?;
        if start >= from && matches_at(start) {
            return Some((start, chars.len(), chars.len()));
        }
        return None;
    }
    // Unanchored: one character must precede and follow the pattern, and
    // the whole match (that leading character included) must start at or
    // after `from`.
    let first = from.saturating_add(1).max(1);
    let last = chars.len().checked_sub(plen + 1)?;
    let mut start = first.max(1);
    while start <= last {
        if matches_at(start) {
            return Some((start, start + plen, start + plen + 1));
        }
        start += 1;
    }
    None
}
