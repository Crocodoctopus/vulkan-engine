use ash::vk;

pub const fn const_max<const N: usize>(values: [usize; N]) -> usize {
    let mut i = 0;
    let mut max = 0;

    while i < N {
        if values[i] > max {
            max = values[i];
        }
        i += 1;
    }

    max
}

pub const fn const_min<const N: usize>(values: [usize; N]) -> usize {
    let mut i = 0;
    let mut min = usize::MAX;

    while i < N {
        if values[i] < min {
            min = values[i];
        }
        i += 1;
    }

    min
}

pub fn format_usize_commas(value: usize) -> String {
    let s = value.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);

    let mut first_group = s.len() % 3;
    if first_group == 0 && !s.is_empty() {
        first_group = 3;
    }

    if !s.is_empty() {
        out.push_str(&s[..first_group]);
        let mut i = first_group;
        while i < s.len() {
            out.push(',');
            out.push_str(&s[i..i + 3]);
            i += 3;
        }
    }

    out
}

pub fn format_bytes(bytes: usize) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];

    let mut value = bytes as f64;
    let mut unit = 0usize;

    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }

    if unit == 0 { format!("{bytes} {}", UNITS[unit]) } else { format!("{value:.1} {}", UNITS[unit]) }
}

// Vulkan's wait any for timelines is unreliable.
pub(crate) unsafe fn wait_semaphores_any_fallback(
    device: &ash::Device,
    semaphores: &[vk::Semaphore],
    values: &[u64],
) -> Result<(), vk::Result> {
    debug_assert_eq!(semaphores.len(), values.len());

    loop {
        for i in 0..semaphores.len() {
            if device.get_semaphore_counter_value(semaphores[i])? >= values[i] {
                return Ok(());
            }
        }

        std::hint::spin_loop();
        std::thread::yield_now();
    }
}
