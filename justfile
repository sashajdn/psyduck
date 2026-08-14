set dotenv-load := true

modal_rust_toolchain := "1.96.0"
modal_uv_cache := "./deployment/modal/.cache/uv"
modal_uv := "./deployment/modal/.tools/bin/uv"
modal_uv_version := "0.12.3"

metrics:
    docker compose up --detach

# Install pinned local tooling and create the locked Python environment.
configure-modal:
    #!/usr/bin/env bash
    set -euo pipefail

    uv_bin="{{ modal_uv }}"
    expected_version="{{ modal_uv_version }}"
    installed_version=""

    if [[ -x "${uv_bin}" ]]; then
        installed_version="$("${uv_bin}" --version | awk '{print $2}')"
    fi

    if [[ "${installed_version}" != "${expected_version}" ]]; then
        install_dir="$(pwd)/deployment/modal/.tools/bin"
        installer="$(mktemp -t psyduck-uv-installer.XXXXXX)"
        trap 'rm -f "${installer}"' EXIT

        mkdir -p "${install_dir}"
        curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error \
            "https://astral.sh/uv/${expected_version}/install.sh" \
            --output "${installer}"
        UV_INSTALL_DIR="${install_dir}" UV_NO_MODIFY_PATH=1 sh "${installer}"
    fi

    UV_CACHE_DIR="{{ modal_uv_cache }}" \
        "${uv_bin}" sync --project deployment/modal --locked --all-groups
    UV_CACHE_DIR="{{ modal_uv_cache }}" \
        "${uv_bin}" run --project deployment/modal --locked modal --version

# Re-synchronize the environment after dependency changes.
modal-sync: configure-modal

# Run any Rust binary on an explicitly selected Modal GPU.
modal-run rust_bin gpu:
    #!/usr/bin/env bash
    set -euo pipefail

    if [[ -n "${PSYDUCK_MODAL_TOKEN_ID:-}" ]]; then
        export MODAL_TOKEN_ID="${PSYDUCK_MODAL_TOKEN_ID}"
    fi
    if [[ -n "${PSYDUCK_MODAL_TOKEN_SECRET:-}" ]]; then
        export MODAL_TOKEN_SECRET="${PSYDUCK_MODAL_TOKEN_SECRET}"
    fi
    if [[ -n "${PSYDUCK_MODAL_PROFILE:-}" ]]; then
        export MODAL_PROFILE="${PSYDUCK_MODAL_PROFILE}"
    fi

    export PSYDUCK_MODAL_RUST_BIN="{{ rust_bin }}"
    export PSYDUCK_MODAL_GPU="{{ gpu }}"
    export PSYDUCK_MODAL_RUST_TOOLCHAIN="{{ modal_rust_toolchain }}"
    UV_CACHE_DIR="{{ modal_uv_cache }}" \
        {{ modal_uv }} run --project deployment/modal --locked \
        modal run deployment/modal/psyduck_modal/app.py

# Validate matrix-add parity on a Modal GPU.
modal-matrix-add-validate:
    just modal-run gpu_add_validate T4

modal-format:
    UV_CACHE_DIR="{{ modal_uv_cache }}" \
        {{ modal_uv }} run --project deployment/modal --locked ruff format deployment/modal

modal-check:
    UV_CACHE_DIR="{{ modal_uv_cache }}" \
        {{ modal_uv }} run --project deployment/modal --locked ruff format --check deployment/modal
    UV_CACHE_DIR="{{ modal_uv_cache }}" \
        {{ modal_uv }} run --project deployment/modal --locked ruff check deployment/modal
