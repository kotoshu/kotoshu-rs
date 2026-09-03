//! C ABI for Kotoshu — always available.
//!
//! Entry point for hosts without a dedicated binding (Go via cgo, Python
//! via ctypes, the conformance runner). Dedicated bindings (Ruby, WASM) call
//! [`crate::ffi::shared`] directly but must stay byte-compatible with this
//! surface: the conformance vectors hold all of them to identical outputs.
//!
//! # Memory management
//!
//! Buffers returned by [`kotoshu_batch`] are heap-allocated; the caller
//! frees them with [`kotoshu_free`].

use std::ffi::{c_char, c_int};

use crate::ffi::shared::{self, Status};

/// NUL-terminated crate version (e.g. `"0.1.0"`). Static; do not free.
#[unsafe(no_mangle)]
pub extern "C" fn kotoshu_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// Execute one batch request.
///
/// Decodes `input` (`input_len` bytes, [`shared`] wire format), produces a
/// response, and writes it to `*output` / `*output_len`. The caller owns the
/// output buffer and frees it with [`kotoshu_free`].
///
/// Returns a [`Status`] code as `c_int`; `0` = [`Status::Ok`].
///
/// TODO(P1/P2): the P0 response is [`shared::stub_response`] — shape-correct,
/// engine-empty.
///
/// # Safety
///
/// `input` must point to `input_len` readable bytes for the duration of the
/// call; `output`/`output_len` must point to valid writable slots.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kotoshu_batch(
    input: *const u8,
    input_len: usize,
    output: *mut *mut u8,
    output_len: *mut usize,
) -> c_int {
    if input.is_null() || output.is_null() || output_len.is_null() {
        return Status::NullPointer as c_int;
    }
    let request_bytes = unsafe { std::slice::from_raw_parts(input, input_len) };
    let request = match shared::decode_request(request_bytes) {
        Ok(request) => request,
        Err(status) => return status as c_int,
    };
    let buffer = shared::encode_response(&shared::stub_response(&request)).into_boxed_slice();
    unsafe {
        *output_len = buffer.len();
        *output = Box::into_raw(buffer).cast();
    }
    Status::Ok as c_int
}

/// Free a buffer returned by [`kotoshu_batch`].
///
/// # Safety
///
/// `ptr`/`len` must be exactly the values written by [`kotoshu_batch`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kotoshu_free(ptr: *mut u8, len: usize) {
    unsafe {
        if !ptr.is_null() {
            drop(Vec::from_raw_parts(ptr, len, len));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::shared::Request;

    #[test]
    fn version_is_nul_terminated() {
        let version = unsafe { std::ffi::CStr::from_ptr(kotoshu_version()) }
            .to_str()
            .unwrap();
        assert_eq!(version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn batch_round_trips_a_check_request() {
        let request = Request::Check {
            language: "en".to_owned(),
            words: vec!["hello".to_owned(), "recieve".to_owned()],
        };
        let input = shared::encode_request(&request);

        let mut output: *mut u8 = std::ptr::null_mut();
        let mut output_len: usize = 0;
        let status =
            unsafe { kotoshu_batch(input.as_ptr(), input.len(), &mut output, &mut output_len) };
        assert_eq!(status, Status::Ok as c_int);

        let bytes = unsafe { std::slice::from_raw_parts(output, output_len) };
        let response = shared::decode_response(bytes).unwrap();
        assert_eq!(
            response,
            shared::Response::Check {
                correct: vec![false, false]
            }
        );
        unsafe { kotoshu_free(output, output_len) };
    }

    #[test]
    fn batch_rejects_null_pointer() {
        assert_eq!(
            unsafe {
                kotoshu_batch(
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            },
            Status::NullPointer as c_int
        );
    }
}
