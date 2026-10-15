use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

// If you set the environment variable NKSYNTH_GIT_HASH, we will use that as the git hash for
// --version.
//
// Otherwise, we will run git.
fn main() {
    println!("cargo:rerun-if-env-changed=NKSYNTH_GIT_HASH");
    let revision = std::env::var("NKSYNTH_GIT_HASH")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            let hash = git(&["rev-parse", "HEAD"])?;
            for reference in ["HEAD", "index"] {
                if let Some(path) = git(&["rev-parse", "--git-path", reference]) {
                    println!("cargo:rerun-if-changed={path}");
                }
            }
            if let Some(reference) = git(&["symbolic-ref", "-q", "HEAD"])
                && let Some(path) = git(&["rev-parse", "--git-path", &reference])
            {
                println!("cargo:rerun-if-changed={path}");
            }
            let dirty = !git(&["status", "--porcelain", "--untracked-files=no"])?.is_empty();
            Some(format!("{hash}{}", if dirty { "-dirty" } else { "" }))
        })
        .unwrap_or_else(|| "unknown".to_owned());
    // Recompute the dirty marker when build inputs change as well as on commits.
    for path in ["build.rs", "Cargo.toml", "Cargo.lock", "src"] {
        println!("cargo:rerun-if-changed={path}");
    }
    println!(
        "cargo:rustc-env=NKSYNTH_VERSION={} (git {revision})",
        env!("CARGO_PKG_VERSION")
    );
}
