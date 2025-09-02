#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct fasttext_t {
  _unused: [u8; 0],
}

unsafe impl Send for fasttext_t {}
unsafe impl Sync for fasttext_t {}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct fasttext_prediction_t {
  pub probability: f32,
  pub label: *mut std::os::raw::c_char,
}

unsafe impl Send for fasttext_prediction_t {}
unsafe impl Sync for fasttext_prediction_t {}

unsafe extern "C" {
  pub fn fasttext_delete(ft: *mut fasttext_t);

  pub fn fasttext_free_predictions(
    predictions: *mut fasttext_prediction_t,
    n_predictions: std::os::raw::c_int,
  );

  pub fn fasttext_load_model(ft: *mut fasttext_t, path: *const std::os::raw::c_char);

  pub fn fasttext_load_model_from_buffer(
    ft: *mut fasttext_t,
    data: *const std::os::raw::c_void,
    size: usize,
  );

  pub fn fasttext_new() -> *mut fasttext_t;

  pub fn fasttext_predict(
    ft: *mut fasttext_t,
    text: *const std::os::raw::c_char,
    k: i32,
    threshold: f32,
    n_predictions: *mut std::os::raw::c_int,
  ) -> *mut fasttext_prediction_t;
}
