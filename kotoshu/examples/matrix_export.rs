//! Plan 136: export a language's derived typo-retrieval matrix as a
//! shippable artifact, so first arming is a download instead of a
//! ~25 s derivation. Byte format (little-endian, versioned):
//!
//! ```text
//! magic     4  b"KTM1"
//! version   u32 = 1
//! count     u32  vocabulary rows
//! dims      u32 = 256
//! rows      count × dims × i8   row-major quantized embeddings
//! scales    count × f32         one per row (max|v| / 127)
//! ```
//!
//! The rows are index-parallel to the fastText FULL tier vocabulary
//! the engine derives from (kotoshu://models/{lang}/full) — the
//! artifact pairs with exactly that vocab, sha-recorded by the
//! registry at generation time.
//!
//! Usage:
//! ```text
//! cargo run -p kotoshu --features model --release --example matrix_export -- \
//!     typo.biencoder.onnx typo.biencoder.vocab.json \
//!     fasttext.en.onnx fasttext.en.vocab.json en.ktm1
//! ```

use std::io::Write;

use kotoshu::rerank::int8_model::Int8Model;
use kotoshu::typo::{OUT_DIM, TypoIndex, TypoModel};

const MAGIC: &[u8; 4] = b"KTM1";
const VERSION: u32 = 1;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        eprintln!(
            "usage: matrix_export <typo.onnx> <typo.vocab.json> <tier.onnx> <tier.vocab.json> <out.ktm1>"
        );
        std::process::exit(2);
    }
    let read = |i: usize| std::fs::read(&args[i]).expect("readable input");
    let typo = TypoModel::parse(&read(1), &read(2)).expect("parse bi-encoder");
    let tier = Int8Model::parse(&read(3), &read(4)).expect("parse tier");

    let index = TypoIndex::build(&typo, tier.vocab());
    let count = index.len() as u32;
    if count == 0 {
        eprintln!("empty index");
        std::process::exit(1);
    }

    let mut out: Vec<u8> = Vec::with_capacity(8 + 12 + count as usize * (OUT_DIM + 4));
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&(OUT_DIM as u32).to_le_bytes());
    index.write_rows(&mut out);

    let path = &args[5];
    let mut file = std::fs::File::create(path).expect("create output");
    file.write_all(&out).expect("write output");
    println!(
        "wrote {path}: {count} rows x {OUT_DIM} ({} bytes)",
        out.len()
    );
}
