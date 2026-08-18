#!/usr/bin/env bash

set -euo pipefail

if (( $# == 0 )); then
    echo "remote execution requires a command" >&2
    exit 2
fi
if [[ ! -f .psyduck-release ]]; then
    echo "release metadata is missing" >&2
    exit 2
fi

release_commit=""
release_dirty=""
while IFS='=' read -r name value; do
    case "${name}" in
        PSYDUCK_GIT_COMMIT) release_commit="${value}" ;;
        PSYDUCK_GIT_DIRTY) release_dirty="${value}" ;;
    esac
done < .psyduck-release

if [[ ! "${release_commit}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "release commit metadata is invalid" >&2
    exit 2
fi
if [[ "${release_dirty}" != "true" && "${release_dirty}" != "false" ]]; then
    echo "release dirty metadata is invalid" >&2
    exit 2
fi

release_root="$(pwd -P)"
remote_root="$(cd "${release_root}/../.." && pwd -P)"
export PATH="${HOME}/.cargo/bin:${PATH}"
export CARGO_TARGET_DIR="${remote_root}/target"
export PSYDUCK_GIT_COMMIT="${release_commit}"
export PSYDUCK_GIT_DIRTY="${release_dirty}"

exec "$@"
