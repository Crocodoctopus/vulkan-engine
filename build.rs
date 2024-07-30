use std::process::Command;

fn main() {
    // Build vertex shader.
    let output = Command::new("glslc")
        .arg("resources/shaders/shader.vert")
        .args(["-o", "src/shader.vert.spirv"])
        .output()
        .unwrap();
    if !output.status.success() {
        panic!("{}", String::from_utf8_lossy(&output.stderr));
    }
    println!("cargo:rerun-if-changed=resources/shaders/shader.vert");
    println!("cargo:rerun-if-changed=src/shader.vert.spirv");

    // Build fragment shader.
    let output = Command::new("glslc")
        .arg("resources/shaders/shader.frag")
        .args(["-o", "src/shader.frag.spirv"])
        .output()
        .unwrap();
    if !output.status.success() {
        panic!("{}", String::from_utf8_lossy(&output.stderr));
    }
    println!("cargo:rerun-if-changed=resources/shaders/shader.frag");
    println!("cargo:rerun-if-changed=src/shader.frag.spirv");
}
