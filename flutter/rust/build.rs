fn main() {
  // Build fasttext C++
  let mut build = cc::Build::new();
  let compiler = build.get_compiler();

  if compiler.is_like_msvc() {
    // Enable exception for clang-cl
    // https://learn.microsoft.com/en-us/cpp/build/reference/eh-exception-handling-model?redirectedfrom=MSDN&view=msvc-170
    build.flag("/EHsc");
  }

  build
    .cpp(true)
    .std("c++17")
    .files([
      "../../src/args.cc",
      "../../src/c_api.cc",
      "../../src/autotune.cc",
      "../../src/matrix.cc",
      "../../src/dictionary.cc",
      "../../src/loss.cc",
      "../../src/productquantizer.cc",
      "../../src/densematrix.cc",
      "../../src/quantmatrix.cc",
      "../../src/vector.cc",
      "../../src/model.cc",
      "../../src/utils.cc",
      "../../src/meter.cc",
      "../../src/fasttext.cc",
    ])
    .includes(["../../src"])
    .flag_if_supported("-pthread")
    .flag_if_supported("-funroll-loops")
    .compile("fasttext-static");

  // Tell Cargo to re-run if there are any changes to the C++ source files
  match std::fs::read_dir("../../src") {
    Ok(files) => {
      for file_result in files {
        match file_result {
          Ok(file) => {
            println!("cargo:rerun-if-changed={}", file.path().display());
          }
          Err(e) => {
            eprintln!("Error reading directory entry: {}", e);
          }
        }
      }
    }
    Err(e) => {
      eprintln!("Error reading C++ source directory: {}", e);
    }
  }
}
