#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unused)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Add a way to easily check if a ${_}Result struct has an error.
pub trait HasError {
  type ResultType;

  fn error(&self) -> *const std::os::raw::c_char;
  fn result(&self) -> Self::ResultType;
}

macro_rules! impl_has_error {
  ($t:ty, $res:ty) => {
    impl HasError for $t {
      type ResultType = $res;

      fn error(&self) -> *const std::os::raw::c_char { self.error }
      fn result(&self) -> Self::ResultType { self.result }
    }
  };
}

impl_has_error!(FastTextResult, *mut fasttext_t);
impl_has_error!(VoidResult, *mut std::os::raw::c_void);
impl_has_error!(FastTextPredictionResult, *mut fasttext_prediction_t);
impl_has_error!(FloatCharPairResult, *mut fasttext_float_char_pair_t);
impl_has_error!(Int32Result, i32);
impl_has_error!(IntResult, std::os::raw::c_int);
