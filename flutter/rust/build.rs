use std::{env, thread};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn build_wasm() {
  // 1. List all C++ source files.
  let files = vec![
    // "src/args.cc",
    // "src/autotune.cc",
    // "src/matrix.cc",
    // "src/dictionary.cc",
    // "src/loss.cc",
    // "src/productquantizer.cc",
    // "src/densematrix.cc",
    // "src/quantmatrix.cc",
    // "src/vector.cc",
    // "src/model.cc",
    // "src/utils.cc",
    // "src/meter.cc",
    // "src/fasttext.cc",
    // "webassembly/fasttext_wasm.cc",

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
  ];

  // 2. Define the compiler flags from EMCXXFLAGS.
  let flags = vec![
    "--bind",
    "--std=c++11",
    "-Isrc/",
    "-sWASM=1",
    "-sALLOW_MEMORY_GROWTH=1",
    "-sEXPORTED_RUNTIME_METHODS=['addOnPostRun', 'FS']",
    "-sDISABLE_EXCEPTION_CATCHING=0",
    "-sEXCEPTION_DEBUG=1",
    "-sFORCE_FILESYSTEM=1",
    "-sMODULARIZE=1",
    "-sEXPORT_ES6=1",
    r#"-sEXPORT_NAME="FastTextModule""#,
  ];

  // 3. Set the output file path.
  // Ensure the output directory exists.
  let out_dir = "webassembly";
  std::fs::create_dir_all(out_dir).expect("Failed to create output directory");
  let out_file = Path::new(out_dir).join("fasttext_wasm.js");

  // 4. Execute the em++ command.
  let mut child = Command::new("em++.bat")
    .args(&files)
    .args(&flags)
    .arg("-o")
    .arg(&out_file)
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .spawn()
    .expect("Failed to execute em++ command. Is Emscripten SDK in your PATH?");

  // Take stdout and stderr handles
  let stdout = child.stdout.take().expect("Failed to capture stdout");
  let stderr = child.stderr.take().expect("Failed to capture stderr");

  // Spawn a thread to handle stdout
  let stdout_thread = thread::spawn(move || {
    let reader = BufReader::new(stdout);
    for line in reader.lines() {
      if let Ok(line_content) = line {
        println!("cargo::warning=[em++] {}", line_content);
      }
    }
  });

  // Spawn a thread to handle stderr
  let stderr_thread = thread::spawn(move || {
    let reader = BufReader::new(stderr);
    for line in reader.lines() {
      if let Ok(line_content) = line {
        println!("cargo::warning=[em++] {}", line_content);
      }
    }
  });

  // Wait for both reader threads to finish
  stdout_thread.join().expect("stdout thread panicked");
  stderr_thread.join().expect("stderr thread panicked");

  let status = child.wait().expect("Failed to execute em++ command. Is Emscripten SDK in your PATH?");

  // Ensure the build was successful.
  if !status.success() {
    panic!("em++ build failed with status: {}", status);
  }

  // 5. Tell Cargo when to re-run this script.
  // Re-run if build.rs changes or any of the C++ source files change.
  println!("cargo:rerun-if-changed=build.rs");
  for file in &files {
    println!("cargo:rerun-if-changed={}", file);
  }

  // println!("cargo::rustc-link-arg=-sMAIN_MODULE=1");
}

fn main() {
  // The bindgen::Builder is the main entry point
  // to bindgen, and lets you build up options for
  // the resulting bindings.
  let bindings = bindgen::Builder::default()
    // .clang_arg("--target=x86_64-pc-windows-msvc")
    // The input header we would like to generate
    // bindings for.
    .header("../../src/c_api.h")
    // Tell cargo to invalidate the built crate whenever any of the
    // included header files changed.
    .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
    // Finish the builder and generate the bindings.
    .generate()
    // Unwrap the Result and panic on failure.
    .expect("Unable to generate bindings");

  // Write the bindings to the $OUT_DIR/bindings.rs file.
  let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
  bindings
    .write_to_file(out_path.join("bindings.rs"))
    .expect("Couldn't write bindings!");

  // build_wasm();

  // Build fasttext C++
  let mut build = cc::Build::new();
  let compiler = build.compiler("em++.bat").get_compiler();

  if compiler.is_like_msvc() {
    // Enable exception for clang-cl
    // https://learn.microsoft.com/en-us/cpp/build/reference/eh-exception-handling-model?redirectedfrom=MSDN&view=msvc-170
    build.flag("/EHsc");
  }

  // build
  //   .cpp(true)
  //   .std("c++17")
  //   .files([
  //     "../../src/args.cc",
  //     "../../src/c_api.cc",
  //     "../../src/autotune.cc",
  //     "../../src/matrix.cc",
  //     "../../src/dictionary.cc",
  //     "../../src/loss.cc",
  //     "../../src/productquantizer.cc",
  //     "../../src/densematrix.cc",
  //     "../../src/quantmatrix.cc",
  //     "../../src/vector.cc",
  //     "../../src/model.cc",
  //     "../../src/utils.cc",
  //     "../../src/meter.cc",
  //     "../../src/fasttext.cc",
  //   ])
  //   .includes(["../../src"])
  //   .flag_if_supported("-pthread")
  //   .flag_if_supported("-funroll-loops")
  //   .compile("fasttext-static");

  build
    .target("wasm32-unknown-emscripten")
    .cpp(true)
    // .flag("--bind")
    .std("c++11")
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
      // "../../webassembly/fasttext_wasm.cc",
    ])
  // src/args.cc
  // src/autotune.cc
  // src/matrix.cc
  // src/dictionary.cc
  // src/loss.cc
  // src/productquantizer.cc
  // src/densematrix.cc
  // src/quantmatrix.cc
  // src/vector.cc
  // src/model.cc
  // src/utils.cc
  // src/meter.cc
  // src/fasttext.cc
  // webassembly/fasttext_wasm.cc
    .includes(["../../src"])
    // .flag("-sSTANDALONE_WASM=0")
    // .flag("-sERROR_ON_UNDEFINED_SYMBOLS=0")
    // .flag("-sWASM=1")
    .flag("-fPIC") // Need position-independent code for linking rust code
    // .flag("-sWASM=1")
    // .flag("-sALLOW_MEMORY_GROWTH=1")
    // .flag("-sEXPORTED_RUNTIME_METHODS=['addOnPostRun', 'FS']")
    .flag("-fexceptions")
    .flag("-sDISABLE_EXCEPTION_CATCHING=0")
    // .flag("-sEXCEPTION_DEBUG=1")
    // .flag("-sFORCE_FILESYSTEM=1")
    // .flag("-sMODULARIZE=1")
    // .flag("-sEXPORT_ES6=1")
    // .flag("-sEXPORT_NAME=\"FastTextModule\"")
    // .flag("-fexceptions") // enable C++ exception handling
    // .flag("-sDISABLE_EXCEPTION_CATCHING=0") // allow try/catch
    // .flag("--no-entry")
    .compile("fasttext");

  // WASM
  // ALLOW_MEMORY_GROWTH
  // EXPORTED_RUNTIME_METHODS
  // EXCEPTION_DEBUG
  // FORCE_FILESYSTEM
  // MODULARIZE
  // EXPORT_ES6
  // EXPORT_NAME

//   em++ --bind --std=c++11 -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s "EXPORTED_RUNTIME_METHODS=['addOnPostRun', 'FS']"
  // -s "DISABLE_EXCEPTION_CATCHING=0" -s "EXCEPTION_DEBUG=1" -s "FORCE_FILESYSTEM=1" -s "MODULARIZE=1"
  // -s "EXPORT_ES6=1" -s 'EXPORT_NAME="FastTextModule"' -Isrc/
  // src/args.cc src/autotune.cc src/matrix.cc src/dictionary.cc src/loss.cc src/productquantizer.cc
  // src/densematrix.cc src/quantmatrix.cc src/vector.cc src/model.cc src/utils.cc src/meter.cc
  // src/fasttext.cc webassembly/fasttext_wasm.cc -o webassembly/fasttext_wasm.js

  // --bind
  //   --std=c++11
  //   -s WASM=1
  //   -s ALLOW_MEMORY_GROWTH=1
  //   -s "EXPORTED_RUNTIME_METHODS=['addOnPostRun', 'FS']"
  //   -s "DISABLE_EXCEPTION_CATCHING=0"
  //   -s "EXCEPTION_DEBUG=1"
  //   -s "FORCE_FILESYSTEM=1"
  //   -s "MODULARIZE=1"
  //   -s "EXPORT_ES6=1"
  //   -s 'EXPORT_NAME="FastTextModule"'

  println!("cargo::rustc-link-arg=-sMAIN_MODULE=1");

  // println!("cargo::rustc-link-arg=-sSTANDALONE_WASM=1");
  // println!("cargo::rustc-link-arg=-sMODULARIZE=1");
  // println!("cargo::rustc-link-arg=-sEXPORT_ES6=1");
}
