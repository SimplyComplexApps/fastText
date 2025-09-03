use fasttext::api::fasttext::FastText;

#[test]
fn test_fasttext_load_model() {
    let mut fasttext = FastText::new().expect("Failed to create FastText instance");
    assert!(fasttext
      .load_model("tests/fixtures/cooking.model.bin")
      .is_ok());
}

// TODO: Need to catch the error in C/C++ side before crossing the FFI boundary
#[test]
fn test_fasttext_load_invalid_model() {
    let mut fasttext = FastText::new().expect("Failed to create FastText instance");
    let result = fasttext.load_model("tests/fixtures/invalid.model.bin");
    assert!(result.is_err());
    assert_eq!("tests/fixtures/invalid.model.bin has wrong file format!", result.unwrap_err());
}

#[test]
fn test_fasttext_predict() {
    let mut fasttext = FastText::new().expect("Failed to create FastText instance");
    fasttext
        .load_model("tests/fixtures/cooking.model.bin")
        .unwrap();
    let preds = fasttext
        .predict("Which baking dish is best to bake a banana bread ?", 2, 0.0)
        .unwrap();
    assert_eq!(2, preds.len());
    assert_eq!("__label__baking", &preds[0].label);
    assert_eq!("__label__bread", &preds[1].label);
}

#[test]
fn test_fasttext_predict_from_buffer() {
    let model_path_str = "tests/fixtures/cooking.model.bin";
    let mut fasttext = FastText::new().expect("Failed to create FastText instance");
    let model_data = std::fs::read(model_path_str).expect("Failed to read model file");
    fasttext
        .load_model_from_buffer(&model_data)
        .unwrap();
    let preds = fasttext
        .predict("Which baking dish is best to bake a banana bread ?", 2, 0.0)
        .unwrap();
    assert_eq!(2, preds.len());
    assert_eq!("__label__baking", &preds[0].label);
    assert_eq!("__label__bread", &preds[1].label);
}

#[test]
fn test_fasttext_predict_on_language() {
    // The language identification model can be downloaded from:
    // https://fasttext.cc/docs/en/language-identification.html
    let model_path_str = "tests/fixtures/lid.176.ftz";
    if !std::path::Path::new(model_path_str).exists() {
        eprintln!(
            "Skipping test: model file not found at '{}'. Please download it.",
            model_path_str
        );
        return;
    }

    // No more `unsafe` block! The RAII wrapper handles all unsafe operations.
    let mut ft = FastText::new().expect("Failed to create FastText instance");
    ft.load_model(model_path_str)
        .expect("Failed to load model");

    let text = "Which baking dish is best to bake a banana bread?";
    let predictions = ft.predict(text, 2, 0.0).expect("Prediction failed");

    assert_eq!(predictions.len(), 2, "Expected 2 predictions");

    // The predictions are now safe Rust structs.
    assert_eq!(predictions[0].label, "__label__en");
    assert!(
        predictions[0].probability > 0.5 && predictions[0].probability <= 1.0,
        "Probability for 'en' out of range"
    );

    // Check that the second prediction exists and has a valid probability.
    assert_eq!(predictions[1].label, "__label__hu");
    assert!(predictions[1].probability >= 0.0 && predictions[1].probability < predictions[0].probability, "Second probability is invalid");
}

#[test]
fn test_fasttext_nn() {
    let mut fasttext = FastText::new().expect("Failed to create FastText instance");
    fasttext
      .load_model("tests/fixtures/lid.176.ftz")
      .unwrap();
    let neighbors = fasttext
      .get_nn("King", 3)
      .unwrap();
    assert_eq!(3, neighbors.len());
    assert_eq!("University", &neighbors[0].1);
    assert_eq!("city", &neighbors[1].1);
    assert_eq!("won", &neighbors[2].1);
}

#[test]
fn test_fasttext_get_analogies() {
    let mut fasttext = FastText::new().expect("Failed to create FastText instance");
    fasttext
      .load_model("tests/fixtures/lid.176.ftz")
      .unwrap();
    let analogies = fasttext
      .get_analogies(3, "king", "queen", "man")
      .unwrap();
    assert_eq!(3, analogies.len());
    assert_eq!("dallas", &analogies[0].1);
    assert_eq!("temps", &analogies[1].1);
    assert_eq!("Giro", &analogies[2].1);
}

#[test]
fn test_fasttext_get_word_id() {
    let mut fasttext = FastText::new().expect("Failed to create FastText instance");
    fasttext
      .load_model("tests/fixtures/lid.176.ftz")
      .unwrap();
    let id = fasttext
      .get_word_id("king")
      .unwrap();
    assert_eq!(1567, id);
}

#[test]
fn test_fasttext_get_subword_id() {
    let mut fasttext = FastText::new().expect("Failed to create FastText instance");
    fasttext
      .load_model("tests/fixtures/lid.176.ftz")
      .unwrap();
    let id = fasttext
      .get_subword_id("king")
      .unwrap();
    assert_eq!(743833, id);
}

#[test]
fn test_fasttext_get_dimension() {
    let mut fasttext = FastText::new().expect("Failed to create FastText instance");
    fasttext
      .load_model("tests/fixtures/lid.176.ftz")
      .unwrap();
    let dim = fasttext
      .get_dimension()
      .unwrap();
    assert_eq!(16, dim);
}

#[test]
fn test_fasttext_get_word_vector() {
    let mut fasttext = FastText::new().expect("Failed to create FastText instance");
    fasttext
      .load_model("tests/fixtures/lid.176.ftz")
      .unwrap();
    let vec = fasttext
      .get_word_vector("king")
      .unwrap();
    assert_eq!(
        vec![
            -0.23979793, -0.44772798, -0.5446826, 0.34108365, -0.25004652, -0.23136835, 0.1491945,
            0.36125875, -0.26137757, -0.18064976, 0.38271922, 0.67140853, 0.2586546, 0.41923445,
            -0.21165611, 0.49381053
        ],
        vec
    );
}

#[test]
fn test_fasttext_get_sentence_vector() {
    let mut fasttext = FastText::new().expect("Failed to create FastText instance");
    fasttext
      .load_model("tests/fixtures/lid.176.ftz")
      .unwrap();
    let vec = fasttext
      .get_sentence_vector("king is to queen as man is to ?")
      .unwrap();
    assert_eq!(
        vec![
            0.38501006, 0.15357634, -1.5447631, 0.6713988, -1.1916713, -0.23329756, 1.1790949,
            0.22402689, -0.7763773, -1.5958408, 0.7044491, -0.20077848, 1.0720185, 0.27003786,
            -0.36773762, 1.2104399
        ],
        vec
    );
}
