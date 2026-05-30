use std::process::Command;

fn main() {
    // Add shader filenames here to build them.
    let shaders: &[&str] = &[
        "render.vert",
        "render.frag",
        "frustum_cull.comp",
    ];

    for shader in shaders {
        let input = format!("resources/shaders/{shader}");
        let output_path = format!("src/{shader}.spirv");

        let output = Command::new("glslc")
            .arg(&input)
            .args(["-o", &output_path])
            .output()
            .unwrap();
        if !output.status.success() {
            panic!("{}", String::from_utf8_lossy(&output.stderr));
        }

        println!("cargo:rerun-if-changed={input}");
        println!("cargo:rerun-if-changed={output_path}");
    }
}
