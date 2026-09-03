//! Reference shim for the `ruby` feature: the same forwarding `init` the
//! kotoshu gem's `ext/kotoshu_native` will contain (plan 66). This crate
//! exists so the shim pattern itself is what the smoke test exercises —
//! see `scripts/ruby_ffi_smoke.sh` and `tests/ruby_ffi_smoke.rb`.

use magnus::{Error, Ruby};

#[magnus::init]
fn init(ruby: &Ruby) -> Result<(), Error> {
    kotoshu::ffi::ruby::init(ruby)
}
