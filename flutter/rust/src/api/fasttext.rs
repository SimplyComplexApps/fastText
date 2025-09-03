use fasttext_bindings::bindings::{
  fasttext_delete, fasttext_free_predictions, fasttext_load_model, fasttext_new, fasttext_predict,
  fasttext_load_model_from_buffer, fasttext_t, fasttext_get_nn, fasttext_free_float_char_pair,
  fasttext_get_analogies, fasttext_get_word_id, fasttext_get_subword_id, fasttext_save_model,
  fasttext_get_dimension, fasttext_get_word_vector, fasttext_get_sentence_vector, HasError
};
use std::ffi::{c_void, CStr, CString};
use flutter_rust_bridge::frb;

fn handle_result<T: HasError>(result: T) -> Result<T::ResultType, String> {
  let error = result.error();
  if error.is_null() {
    Ok(result.result())
  } else {
    let err = unsafe { CStr::from_ptr(error).to_string_lossy().into_owned() };
    Err(err)
  }
}

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
    let result = unsafe { fasttext_new() };
    let handle = handle_result(result)?;

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
    let result = unsafe { fasttext_load_model(self.handle, c_path.as_ptr()) };
    handle_result(result)?;

    Ok(())
  }

  /// Loads a model from a buffer.
  ///
  /// # Arguments
  ///
  /// * `buffer` - A byte slice containing the model data.
  pub fn load_model_from_buffer(&mut self, buffer: &[u8]) -> Result<(), String> {
    // This is safe because we've checked the handle is not null on creation.
    let result = unsafe {
      fasttext_load_model_from_buffer(self.handle, buffer.as_ptr() as *const c_void, buffer.len())
    };
    handle_result(result)?;
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

    let mut n_predictions: usize = 0;

    // The C API allocates memory for the predictions, which we must free.
    let result = unsafe {
      fasttext_predict(self.handle, c_text.as_ptr(), k, threshold, &mut n_predictions)
    };
    let preds_ptr = handle_result(result)?;

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
        std::slice::from_raw_parts(preds_ptr, n_predictions);
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

  /// Nearest neighbors for a given word.
  ///
  /// # Arguments
  ///
  /// * `word` - The word to find nearest neighbors for.
  /// * `k` - The number of nearest neighbors to return.
  pub fn get_nn(&self, word: &str, k: i32) -> Result<Vec<(f32, String)>, String> {
    let c_word = CString::new(word)
      .map_err(|e| format!("Failed to create CString from word: {}", e))?;

    let mut n_neighbors: usize = 0;

    // The C API allocates memory for the predictions, which we must free.
    let result = unsafe {
      fasttext_get_nn(self.handle, c_word.as_ptr(), k, &mut n_neighbors)
    };
    let neighbors_ptr = handle_result(result)?;

    if neighbors_ptr.is_null() {
      return if n_neighbors == 0 {
        Ok(Vec::new())
      } else {
        Err(
          "fasttext_get_nn returned a null pointer but a non-zero prediction count."
            .to_string(),
        )
      };
    }

    // Create a safe slice, copy the data into a safe Rust Vec,
    // and then immediately free the C-allocated memory.
    let rust_neighbors = unsafe {
      let neighbors_slice = std::slice::from_raw_parts(neighbors_ptr, n_neighbors);
      let result: Vec<(f32, String)> = neighbors_slice
        .iter()
        .map(|n| {
          let word_cstr = CStr::from_ptr(n.second);
          (n.first, word_cstr.to_string_lossy().into_owned())
        })
        .collect();
      fasttext_free_float_char_pair(neighbors_ptr, n_neighbors);
      result
    };

    Ok(rust_neighbors)
  }

  /// Solves the word analogy problem.
  ///
  /// Solve word analogy problems of the form "A is to B as C is to ?".
  /// For example, "king is to queen as man is to ?". The goal is to find the word that best fits
  /// the question mark (in this case, "woman").
  ///
  /// # Arguments
  ///
  /// * `k` - The number of analogies to return.
  /// * `wordA` - The word A in "A is to B as C is to ?".
  /// * `wordB` - The word B in "A is to B as C is to ?".
  /// * `wordC` - The word C in "A is to B as C is to ?".
  pub fn get_analogies(&self, k: i32, word_a: &str, word_b: &str, word_c: &str,) -> Result<Vec<(f32, String)>, String> {
    let c_word_a = CString::new(word_a)
      .map_err(|e| format!("Failed to create CString from word_a: {}", e))?;
    let c_word_b = CString::new(word_b)
      .map_err(|e| format!("Failed to create CString from word_b: {}", e))?;
    let c_word_c = CString::new(word_c)
      .map_err(|e| format!("Failed to create CString from word_c: {}", e))?;

    let mut n_analogies: usize = 0;

    // The C API allocates memory for the predictions, which we must free.
    let result = unsafe {
      fasttext_get_analogies(self.handle, k, c_word_a.as_ptr(), c_word_b.as_ptr(), c_word_c.as_ptr(), &mut n_analogies)
    };
    let analogies_ptr = handle_result(result)?;

    if analogies_ptr.is_null() {
      return if n_analogies == 0 {
        Ok(Vec::new())
      } else {
        Err(
          "get_analogies returned a null pointer but a non-zero prediction count."
            .to_string(),
        )
      };
    }

    // Create a safe slice, copy the data into a safe Rust Vec,
    // and then immediately free the C-allocated memory.
    let rust_analogies = unsafe {
      let neighbors_slice = std::slice::from_raw_parts(analogies_ptr, n_analogies);
      let result: Vec<(f32, String)> = neighbors_slice
        .iter()
        .map(|n| {
          let word_cstr = CStr::from_ptr(n.second);
          (n.first, word_cstr.to_string_lossy().into_owned())
        })
        .collect();
      fasttext_free_float_char_pair(analogies_ptr, n_analogies);
      result
    };

    Ok(rust_analogies)
  }

  /// Get the ID of a word.
  ///
  /// # Arguments
  ///
  /// * `word` - The word to get the ID for.
  pub fn get_word_id(&self, word: &str) -> Result<i32, String> {
    let c_word = CString::new(word)
      .map_err(|e| format!("Failed to create CString from word: {}", e))?;

    // This is safe because we've checked the handle is not null on creation,
    // and the CString is valid.
    let result = unsafe { fasttext_get_word_id(self.handle, c_word.as_ptr()) };
    let word_id = handle_result(result)?;

    Ok(word_id)
  }

  /// Get the subword ID of a word.
  ///
  /// # Arguments
  ///
  /// * `word` - The word to get the ID for.
  pub fn get_subword_id(&self, word: &str) -> Result<i32, String> {
    let c_word = CString::new(word)
      .map_err(|e| format!("Failed to create CString from word: {}", e))?;

    // This is safe because we've checked the handle is not null on creation,
    // and the CString is valid.
    let result = unsafe { fasttext_get_subword_id(self.handle, c_word.as_ptr()) };
    let subword_id = handle_result(result)?;

    Ok(subword_id)
  }

  /// Saves a model to the given path.
  ///
  /// # Arguments
  ///
  /// * `path` - The file path of where to save the model.
  pub fn save_model(&mut self, path: &str) -> Result<(), String> {
    let c_path = CString::new(path)
      .map_err(|e| format!("Failed to create CString from path: {}", e))?;

    // This is safe because we've checked the handle is not null on creation,
    // and the CString is valid.
    let result = unsafe { fasttext_save_model(self.handle, c_path.as_ptr()) };
    handle_result(result)?;
    Ok(())
  }

  /// Get dimension of the model.
  pub fn get_dimension(&self) -> Result<i32, String> {
    // This is safe because we've checked the handle is not null on creation
    let result = unsafe { fasttext_get_dimension(self.handle) };
    let dimension = handle_result(result)?;

    Ok(dimension)
  }

  /// Get the vector of a word.
  ///
  /// # Arguments
  ///
  /// * `word` - The word to get the vector for.
  pub fn get_word_vector(&self, word: &str) -> Result<Vec<f32>, String> {
    let c_word = CString::new(word)
      .map_err(|e| format!("Failed to create CString from word: {}", e))?;
    let dim = self.get_dimension()
      .map_err(|e| format!("Failed to get dimension of model: {}", e))?;

    let mut vec: Vec<f32> = vec![0.0; dim as usize];

    // This is safe because we've checked the handle is not null on creation,
    // and the CString is valid.
    let result = unsafe {
      fasttext_get_word_vector(self.handle, c_word.as_ptr(), vec.as_mut_ptr())
    };
    handle_result(result)?;

    Ok(vec)
  }

  /// Get the vector of a sentence.
  ///
  /// # Arguments
  ///
  /// * `text` - The sentence to get the vector for.
  pub fn get_sentence_vector(&self, text: &str) -> Result<Vec<f32>, String> {
    let c_word = CString::new(text)
      .map_err(|e| format!("Failed to create CString from text: {}", e))?;
    let dim = self.get_dimension()
      .map_err(|e| format!("Failed to get dimension of model: {}", e))?;

    let mut vec: Vec<f32> = vec![0.0; dim as usize];

    // This is safe because we've checked the handle is not null on creation,
    // and the CString is valid.
    let result = unsafe {
      fasttext_get_sentence_vector(self.handle, c_word.as_ptr(), vec.as_mut_ptr())
    };
    handle_result(result)?;

    Ok(vec)
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
