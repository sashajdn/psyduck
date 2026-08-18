#!/usr/bin/env bash

set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
cd "${repository_root}"

remote_host="${PSYDUCK_REMOTE_HOST:-}"
remote_directory="${PSYDUCK_REMOTE_DIR:-psyduck}"

if [[ -z "${remote_host}" ]]; then
    echo "PSYDUCK_REMOTE_HOST must name an SSH host or user@host" >&2
    exit 2
fi
if [[ ! "${remote_host}" =~ ^[A-Za-z0-9._@-]+$ ]] || [[ "${remote_host}" == -* ]]; then
    echo "PSYDUCK_REMOTE_HOST contains unsupported characters" >&2
    exit 2
fi
if [[ -z "${remote_directory}" || "${remote_directory}" == /* ]]; then
    echo "PSYDUCK_REMOTE_DIR must be relative to the remote home directory" >&2
    exit 2
fi
if [[ ! "${remote_directory}" =~ ^[A-Za-z0-9._/-]+$ ]]; then
    echo "PSYDUCK_REMOTE_DIR contains unsupported characters" >&2
    exit 2
fi

IFS='/' read -r -a remote_components <<< "${remote_directory}"
for component in "${remote_components[@]}"; do
    if [[ -z "${component}" || "${component}" == "." || "${component}" == ".." ]]; then
        echo "PSYDUCK_REMOTE_DIR cannot contain empty, '.' or '..' components" >&2
        exit 2
    fi
done

commit="$(git rev-parse HEAD)"
short_commit="$(git rev-parse --short=12 HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then
    dirty="true"
    state="dirty"
else
    dirty="false"
    state="clean"
fi

release_id="${short_commit}-${state}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
incoming="${remote_directory}/.incoming/${release_id}"
release="${remote_directory}/releases/${release_id}"

ssh_options=(
    -o BatchMode=yes
    -o PreferredAuthentications=publickey
    -o PasswordAuthentication=no
    -o KbdInteractiveAuthentication=no
)
rsync_ssh="ssh ${ssh_options[*]}"

echo "syncing ${short_commit} (${state}) to ${remote_host}:${release}" >&2

ssh "${ssh_options[@]}" "${remote_host}" \
    "set -eu; mkdir -p -- '${remote_directory}/.incoming' '${remote_directory}/releases' '${remote_directory}/reports' '${remote_directory}/target'; test ! -e '${incoming}'; mkdir -- '${incoming}'"

existing_repository_files() {
    while IFS= read -r -d '' path; do
        if [[ -e "${path}" || -L "${path}" ]]; then
            printf '%s\0' "${path}"
        fi
    done < <(git ls-files --cached --others --exclude-standard -z)
}

existing_repository_files | RSYNC_RSH="${rsync_ssh}" rsync \
    --archive \
    --compress \
    --from0 \
    --files-from=- \
    ./ \
    "${remote_host}:${incoming}/"

ssh "${ssh_options[@]}" "${remote_host}" \
    "set -eu; printf '%s\n' 'PSYDUCK_GIT_COMMIT=${commit}' 'PSYDUCK_GIT_DIRTY=${dirty}' > '${incoming}/.psyduck-release'; mv -- '${incoming}' '${release}'; cd '${remote_directory}'; ln -s 'releases/${release_id}' '.current-${release_id}'; mv -Tf -- '.current-${release_id}' current"

printf '%s\n' "${release}"
