fn main() {
    use std::process::Command;

    Command::new("glslc")
        .arg("resources/shaders/shader.vert")
        .args(["-o", "src/shader.vert.spirv"])
        .output()
        .unwrap();

    Command::new("glslc")
        .arg("resources/shaders/shader.frag")
        .args(["-o", "src/shader.frag.spirv"])
        .output()
        .unwrap();
}
