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

# Run any Rust binary, with optional JSON-encoded arguments, on a selected Modal GPU.
modal-run rust_bin gpu rust_args_json="[]":
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

    if [[ -z "${PSYDUCK_MODAL_GIT_COMMIT:-}" ]]; then
        PSYDUCK_MODAL_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
        export PSYDUCK_MODAL_GIT_COMMIT
    fi
    if [[ -z "${PSYDUCK_MODAL_GIT_DIRTY:-}" ]]; then
        if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
            PSYDUCK_MODAL_GIT_DIRTY="true"
        else
            PSYDUCK_MODAL_GIT_DIRTY="false"
        fi
        export PSYDUCK_MODAL_GIT_DIRTY
    fi

    export PSYDUCK_MODAL_RUST_BIN="{{ rust_bin }}"
    export PSYDUCK_MODAL_RUST_ARGS_JSON='{{ rust_args_json }}'
    export PSYDUCK_MODAL_GPU="{{ gpu }}"
    export PSYDUCK_MODAL_RUST_TOOLCHAIN="{{ modal_rust_toolchain }}"
    UV_CACHE_DIR="{{ modal_uv_cache }}" \
        {{ modal_uv }} run --project deployment/modal --locked \
        modal run deployment/modal/psyduck_modal/app.py

# Validate matrix-add parity on a Modal GPU.
modal-matrix-add-validate:
    just modal-run gpu_add_validate T4

# Run matmul locally on the host or remotely on a Modal T4.
# Omit N and K for a square M x M multiplication.
matmul target="host" m="512" n="" k="" report_dir="" operations="1" warmup="3" sample_every="1":
    #!/usr/bin/env bash
    set -euo pipefail

    target="{{ target }}"
    m="{{ m }}"
    n="{{ n }}"
    k="{{ k }}"
    report_dir="{{ report_dir }}"
    operations="{{ operations }}"
    warmup="{{ warmup }}"
    sample_every="{{ sample_every }}"
    n="${n:-${m}}"
    k="${k:-${m}}"

    for dimension in "${m}" "${n}" "${k}"; do
        case "${dimension}" in
            4|8|16|32|64|128|256|512|1024|2048|4096) ;;
            *)
                echo "unsupported matrix dimension: ${dimension} (expected 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, or 4096)" >&2
                exit 2
                ;;
        esac
    done

    for value in "${operations}" "${sample_every}"; do
        if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
            echo "operations and sample_every must be positive integers" >&2
            exit 2
        fi
    done
    if [[ ! "${warmup}" =~ ^[0-9]+$ ]]; then
        echo "warmup must be a non-negative integer" >&2
        exit 2
    fi

    case "${target}" in
        host)
            rust_args=(
                --target host --m "${m}" --n "${n}" --k "${k}"
                --operations "${operations}" --warmup "${warmup}"
                --sample-every "${sample_every}"
            )
            if [[ -n "${report_dir}" ]]; then
                rust_args+=(--report-dir "${report_dir}")
            fi
            cargo run --release --bin matmul -- "${rust_args[@]}"
            ;;
        device)
            rust_args=(
                --target device --m "${m}" --n "${n}" --k "${k}"
                --operations "${operations}" --warmup "${warmup}"
                --sample-every "${sample_every}"
            )
            if [[ -n "${report_dir}" ]]; then
                rust_args+=(--report-dir "${report_dir}")
            fi
            rust_args_json="$(
                UV_CACHE_DIR="{{ modal_uv_cache }}" \
                    {{ modal_uv }} run --project deployment/modal --locked \
                    python -c 'import json, sys; print(json.dumps(sys.argv[1:]))' \
                    "${rust_args[@]}"
            )"
            just modal-run matmul T4 "${rust_args_json}"
            ;;
        *)
            echo "unsupported matmul target: ${target} (expected host or device)" >&2
            exit 2
            ;;
    esac

modal-format:
    UV_CACHE_DIR="{{ modal_uv_cache }}" \
        {{ modal_uv }} run --project deployment/modal --locked ruff format deployment/modal

modal-check:
    UV_CACHE_DIR="{{ modal_uv_cache }}" \
        {{ modal_uv }} run --project deployment/modal --locked ruff format --check deployment/modal
    UV_CACHE_DIR="{{ modal_uv_cache }}" \
        {{ modal_uv }} run --project deployment/modal --locked ruff check deployment/modal
