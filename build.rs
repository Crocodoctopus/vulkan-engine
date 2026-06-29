use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=resources/shaders/types.h");
    println!("cargo:rerun-if-changed=resources/shaders/util.h");

    // Add shader filenames here to build them.
    let shaders: &[&str] = &[
        "render.vert",
        "render.frag",
        "overdraw.frag",
        "overshade.frag",
        "overdraw_resolve.comp",
        "frustum_cull.comp",
        "build_hzb.comp",
        "occlusion_cull.comp",
    ];

    for shader in shaders {
        let input = format!("resources/shaders/{shader}");
        let output_path = format!("src/{shader}.spirv");

        let output = Command::new("glslc")
            .arg(&input)
            .args(["--target-env=vulkan1.1", "-o", &output_path])
            .output()
            .unwrap();
        if !output.status.success() {
            panic!("{}", String::from_utf8_lossy(&output.stderr));
        }

        println!("cargo:rerun-if-changed={input}");
        println!("cargo:rerun-if-changed={output_path}");
    }
}
