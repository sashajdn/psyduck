"""Modal framework for executing kernels via rust container."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import modal

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REMOTE_REPOSITORY = "/workspace"
RUST_BINARY_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def _environment(name: str, default: str) -> str:
    return os.environ.get(f"PSYDUCK_MODAL_{name}", default)


def _positive_integer(name: str, default: int) -> int:
    raw_value = _environment(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"PSYDUCK_MODAL_{name} must be an integer, got {raw_value!r}") from error

    if value <= 0:
        raise ValueError(f"PSYDUCK_MODAL_{name} must be positive, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    app_name: str
    cuda_image: str
    gpu: str
    rust_binary: str
    rust_toolchain: str
    timeout_seconds: int

    @classmethod
    def from_environment(cls) -> Settings:
        rust_binary = _environment("RUST_BIN", "gpu_add_validate")
        if not RUST_BINARY_PATTERN.fullmatch(rust_binary):
            raise ValueError(
                "PSYDUCK_MODAL_RUST_BIN may contain only letters, numbers, underscores, and hyphens"
            )

        return cls(
            app_name=_environment("APP_NAME", "psyduck"),
            cuda_image=_environment("CUDA_IMAGE", "nvidia/cuda:12.8.1-devel-ubuntu24.04"),
            gpu=_environment("GPU", "L4"),
            rust_binary=rust_binary,
            rust_toolchain=_environment("RUST_TOOLCHAIN", "1.96.0"),
            timeout_seconds=_positive_integer("TIMEOUT_SECONDS", 600),
        )


settings = Settings.from_environment()

image = (
    modal.Image.from_registry(settings.cuda_image, add_python="3.12")
    .apt_install("build-essential", "ca-certificates", "curl", "pkg-config")
    .env(
        {
            "CARGO_HOME": "/opt/cargo",
            "PATH": "/opt/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "RUST_BACKTRACE": "1",
            "RUSTUP_HOME": "/opt/rustup",
        }
    )
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        f"| sh -s -- -y --profile minimal --default-toolchain {settings.rust_toolchain}"
    )
    .add_local_file(REPOSITORY_ROOT / "Cargo.toml", f"{REMOTE_REPOSITORY}/Cargo.toml", copy=True)
    .add_local_file(REPOSITORY_ROOT / "Cargo.lock", f"{REMOTE_REPOSITORY}/Cargo.lock", copy=True)
    .add_local_dir(REPOSITORY_ROOT / "bin", f"{REMOTE_REPOSITORY}/bin", copy=True)
    .add_local_dir(REPOSITORY_ROOT / "instrument", f"{REMOTE_REPOSITORY}/instrument", copy=True)
    .add_local_dir(REPOSITORY_ROOT / "kernel", f"{REMOTE_REPOSITORY}/kernel", copy=True)
    .add_local_dir(REPOSITORY_ROOT / "model", f"{REMOTE_REPOSITORY}/model", copy=True)
    .add_local_dir(REPOSITORY_ROOT / "tensor", f"{REMOTE_REPOSITORY}/tensor", copy=True)
    .workdir(REMOTE_REPOSITORY)
    .run_commands(f"cargo build --release --locked --bin {settings.rust_binary}")
)

app = modal.App(settings.app_name, image=image)


@app.function(gpu=settings.gpu, timeout=settings.timeout_seconds)
def run_rust_binary(rust_binary: str, gpu: str) -> dict[str, object]:
    """Run the image's selected Rust binary and return its captured output."""
    executable = Path(REMOTE_REPOSITORY) / "target" / "release" / rust_binary
    completed = subprocess.run(
        [executable],
        check=False,
        capture_output=True,
        text=True,
    )

    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")

    completed.check_returncode()
    return {
        "rust_binary": rust_binary,
        "gpu": gpu,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


@app.local_entrypoint()
def main() -> None:
    """Launch the selected binary and render a stable local result."""
    result = run_rust_binary.remote(settings.rust_binary, settings.gpu)
    print(json.dumps(result, indent=2, sort_keys=True))
