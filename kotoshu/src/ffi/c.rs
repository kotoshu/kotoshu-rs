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
//! frees them with [`kotoshu_free`]. Dictionaries registered through
//! [`kotoshu_dict_load`] are dropped with [`kotoshu_dict_free`].

use std::ffi::{CStr, c_char, c_int};
use std::path::Path;

use crate::ffi::shared::{self, Status};

/// NUL-terminated crate version (e.g. `"0.1.0"`). Static; do not free.
#[unsafe(no_mangle)]
pub extern "C" fn kotoshu_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// Load a Hunspell `.aff`/`.dic` pair and register it under `language`
/// for batch requests ([`kotoshu_batch`]).
///
/// Re-registering a language replaces its dictionary. Paths are
/// NUL-terminated UTF-8.
///
/// Returns a [`Status`] code as `c_int`; on failure the previously
/// registered dictionary (if any) is preserved.
///
/// # Safety
///
/// All three pointers must point to NUL-terminated readable UTF-8 for the
/// duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kotoshu_dict_load(
    language: *const c_char,
    aff_path: *const c_char,
    dic_path: *const c_char,
) -> c_int {
    if language.is_null() || aff_path.is_null() || dic_path.is_null() {
        return Status::NullPointer as c_int;
    }
    let Ok(language) = unsafe { CStr::from_ptr(language) }.to_str() else {
        return Status::InvalidUtf8 as c_int;
    };
    let Ok(aff_path) = unsafe { CStr::from_ptr(aff_path) }.to_str() else {
        return Status::InvalidUtf8 as c_int;
    };
    let Ok(dic_path) = unsafe { CStr::from_ptr(dic_path) }.to_str() else {
        return Status::InvalidUtf8 as c_int;
    };
    match super::registry::register(language, Path::new(aff_path), Path::new(dic_path)) {
        Ok(()) => Status::Ok as c_int,
        // The wire has no channel for load-error detail; hosts surface it
        // by trying again through the Rust/Ruby API. Distinct code so it
        // is not confused with "not loaded".
        Err(_) => shared::STATUS_DICTIONARY_LOAD_FAILED,
    }
}

/// Drop the dictionary registered under `language`.
///
/// Returns [`Status::Ok`] whether or not one was registered.
///
/// # Safety
///
/// `language` must point to NUL-terminated readable UTF-8 for the duration
/// of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kotoshu_dict_free(language: *const c_char) -> c_int {
    if language.is_null() {
        return Status::NullPointer as c_int;
    }
    let Ok(language) = unsafe { CStr::from_ptr(language) }.to_str() else {
        return Status::InvalidUtf8 as c_int;
    };
    super::registry::unregister(language);
    Status::Ok as c_int
}

/// Execute one batch request against the registered dictionaries.
///
/// Decodes `input` (`input_len` bytes, [`shared`] wire format), routes it
/// by its `language` (register one with [`kotoshu_dict_load`] first),
/// produces a response, and writes it to `*output` / `*output_len`. The
/// caller owns the output buffer and frees it with [`kotoshu_free`].
///
/// Returns a [`Status`] code as `c_int`; `0` = [`Status::Ok`]. No output
/// buffer is written on failure.
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
    let response = match shared::respond(&request) {
        Ok(response) => response,
        Err(status) => return status as c_int,
    };
    let buffer = shared::encode_response(&response).into_boxed_slice();
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
    use crate::ffi::shared::{Request, Response, decode_response, encode_request};

    const AFF: &[u8] = b"SET UTF-8\n";
    const DIC: &[u8] = b"2\nhello\nworld\n";

    fn cstring(path: &std::path::Path) -> Vec<u8> {
        let mut bytes = path.as_os_str().as_encoded_bytes().to_vec();
        bytes.push(0);
        bytes
    }

    #[test]
    fn version_is_nul_terminated() {
        let version = unsafe { std::ffi::CStr::from_ptr(kotoshu_version()) }
            .to_str()
            .unwrap();
        assert_eq!(version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn dict_lifecycle_and_batch_end_to_end() {
        let dir = std::env::temp_dir();
        let aff = dir.join("kotoshu-c-test.aff");
        let dic = dir.join("kotoshu-c-test.dic");
        std::fs::write(&aff, AFF).unwrap();
        std::fs::write(&dic, DIC).unwrap();
        let language = cstring(std::path::Path::new("zz-c-test"));
        let aff = cstring(&aff);
        let dic = cstring(&dic);

        // Batch before load fails loudly with DictionaryNotLoaded.
        let request = Request::Check {
            language: "zz-c-test".to_owned(),
            words: vec!["hello".to_owned()],
        };
        let input = encode_request(&request);
        let mut output: *mut u8 = std::ptr::null_mut();
        let mut output_len: usize = 0;
        let status =
            unsafe { kotoshu_batch(input.as_ptr(), input.len(), &mut output, &mut output_len) };
        assert_eq!(status, Status::DictionaryNotLoaded as c_int);
        assert!(output.is_null());

        // Load, then batch check and suggest.
        assert_eq!(
            unsafe {
                kotoshu_dict_load(
                    language.as_ptr().cast(),
                    aff.as_ptr().cast(),
                    dic.as_ptr().cast(),
                )
            },
            Status::Ok as c_int
        );

        let request = Request::Check {
            language: "zz-c-test".to_owned(),
            words: vec!["hello".to_owned(), "helo".to_owned()],
        };
        let input = encode_request(&request);
        let status =
            unsafe { kotoshu_batch(input.as_ptr(), input.len(), &mut output, &mut output_len) };
        assert_eq!(status, Status::Ok as c_int);
        let bytes = unsafe { std::slice::from_raw_parts(output, output_len) };
        assert_eq!(
            decode_response(bytes).unwrap(),
            Response::Check {
                correct: vec![true, false]
            }
        );
        unsafe { kotoshu_free(output, output_len) };

        let request = Request::Suggest {
            language: "zz-c-test".to_owned(),
            word: "helo".to_owned(),
            limit: 5,
        };
        let input = encode_request(&request);
        let mut output: *mut u8 = std::ptr::null_mut();
        let mut output_len: usize = 0;
        let status =
            unsafe { kotoshu_batch(input.as_ptr(), input.len(), &mut output, &mut output_len) };
        assert_eq!(status, Status::Ok as c_int);
        let bytes = unsafe { std::slice::from_raw_parts(output, output_len) };
        let Response::Suggest { suggestions } = decode_response(bytes).unwrap() else {
            panic!("expected suggest response");
        };
        assert_eq!(suggestions[0].word, "hello");
        assert_eq!(suggestions[0].distance, 1);
        assert_eq!(
            suggestions[0].source,
            shared::SuggestionSource::EditDistance
        );
        unsafe { kotoshu_free(output, output_len) };

        // Free, then the language is gone again.
        assert_eq!(
            unsafe { kotoshu_dict_free(language.as_ptr().cast()) },
            Status::Ok as c_int
        );
        let input = encode_request(&Request::Check {
            language: "zz-c-test".to_owned(),
            words: vec!["hello".to_owned()],
        });
        let mut output: *mut u8 = std::ptr::null_mut();
        let mut output_len: usize = 0;
        let status =
            unsafe { kotoshu_batch(input.as_ptr(), input.len(), &mut output, &mut output_len) };
        assert_eq!(status, Status::DictionaryNotLoaded as c_int);
    }

    #[test]
    fn dict_load_reports_missing_file() {
        let language = cstring(std::path::Path::new("zz-c-missing"));
        let missing = cstring(&std::env::temp_dir().join("kotoshu-c-missing.aff"));
        let status = unsafe {
            kotoshu_dict_load(
                language.as_ptr().cast(),
                missing.as_ptr().cast(),
                missing.as_ptr().cast(),
            )
        };
        assert_eq!(status, shared::STATUS_DICTIONARY_LOAD_FAILED);
        unsafe { kotoshu_dict_free(language.as_ptr().cast()) };
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

    #[test]
    fn dict_load_rejects_null_pointer() {
        assert_eq!(
            unsafe { kotoshu_dict_load(std::ptr::null(), std::ptr::null(), std::ptr::null()) },
            Status::NullPointer as c_int
        );
    }
}
