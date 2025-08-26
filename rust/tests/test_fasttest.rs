use fasttext::{FastText};

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
