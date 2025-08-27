use crate::bindings::{
  fasttext_delete, fasttext_free_predictions, fasttext_load_model, fasttext_new,
  fasttext_predict, fasttext_load_model_from_buffer, fasttext_t,
};
use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_int;
use flutter_rust_bridge::frb;

/// A safe Rust wrapper for a fastText model.
///
/// This struct handles the creation and destruction of the underlying
/// `fasttext_t` C object, implementing the RAII pattern. When an instance
/// of `FastText` goes out of scope, its `Drop` implementation is automatically
/// called, ensuring the C++ object is deleted and preventing memory leaks.
pub struct FastText {
  // This is the opaque pointer to the C++ fastText object.
  handle: *mut fasttext_t,
}

/// Represents a single prediction from the fastText model.
#[derive(Debug, PartialEq, Clone)]
pub struct Prediction {
  pub probability: f32,
  pub label: String,
}

impl FastText {
  /// Creates a new fastText instance.
  #[frb(sync)]
  pub fn new() -> Result<Self, String> {
    // unsafe block is necessary to call C functions.
    let handle = unsafe { fasttext_new() };
    if handle.is_null() {
      Err("Failed to create fastText handle.".to_string())
    } else {
      Ok(FastText { handle })
    }
  }

  /// Loads a model from the given path.
  ///
  /// # Arguments
  ///
  /// * `path` - The file path of the model to load.
  pub fn load_model(&mut self, path: &str) -> Result<(), String> {
    let c_path = CString::new(path)
      .map_err(|e| format!("Failed to create CString from path: {}", e))?;

    // This is safe because we've checked the handle is not null on creation,
    // and the CString is valid.
    unsafe {
      fasttext_load_model(self.handle, c_path.as_ptr());
    }
    Ok(())
  }

  /// Loads a model from a buffer.
  ///
  /// # Arguments
  ///
  /// * `buffer` - A byte slice containing the model data.
  pub fn load_model_from_buffer(&mut self, buffer: &[u8]) -> Result<(), String> {
    // This is safe because we've checked the handle is not null on creation.
    unsafe {
      fasttext_load_model_from_buffer(self.handle, buffer.as_ptr() as *const c_void, buffer.len());
    }
    Ok(())
  }

  /// Predicts labels for a given text.
  ///
  /// # Arguments
  ///
  /// * `text` - The input text for prediction.
  /// * `k` - The number of top predictions to return.
  /// * `threshold` - The minimum probability for a prediction to be returned.
  pub fn predict(&self, text: &str, k: i32, threshold: f32) -> Result<Vec<Prediction>, String> {
    let c_text = CString::new(text)
      .map_err(|e| format!("Failed to create CString from text: {}", e))?;

    let mut n_predictions: c_int = 0;

    // The C API allocates memory for the predictions, which we must free.
    let preds_ptr = unsafe {
      fasttext_predict(self.handle, c_text.as_ptr(), k, threshold, &mut n_predictions)
    };

    if preds_ptr.is_null() {
      return if n_predictions == 0 {
        Ok(Vec::new())
      } else {
        Err(
          "fasttext_predict returned a null pointer but a non-zero prediction count."
            .to_string(),
        )
      };
    }

    // Create a safe slice, copy the data into a safe Rust Vec,
    // and then immediately free the C-allocated memory.
    let rust_predictions = unsafe {
      let predictions_slice =
        std::slice::from_raw_parts(preds_ptr, n_predictions as usize);
      let result = predictions_slice
        .iter()
        .map(|p| {
          let label_cstr = CStr::from_ptr(p.label);
          Prediction {
            probability: p.probability,
            label: label_cstr.to_string_lossy().into_owned(),
          }
        })
        .collect();
      fasttext_free_predictions(preds_ptr, n_predictions);
      result
    };

    Ok(rust_predictions)
  }
}

// The Drop trait implementation is the heart of RAII.
// When a `FastText` object goes out of scope, this `drop` method is called automatically.
impl Drop for FastText {
  fn drop(&mut self) {
    // This ensures the C++ object is always deleted, preventing memory leaks.
    if !self.handle.is_null() {
      unsafe {
        fasttext_delete(self.handle);
      }
    }
  }
}

// It's good practice to implement Send and Sync if the underlying C library is thread-safe.
// The fastText C++ library is thread-safe for prediction (`const` methods).
unsafe impl Send for FastText {}
unsafe impl Sync for FastText {}

impl Default for FastText {
  fn default() -> Self {
    Self::new().expect("Failed to create default FastText instance")
  }
}
