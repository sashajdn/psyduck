#!/usr/bin/env bash

set -euo pipefail

if (( $# == 0 )); then
    echo "remote run requires a command" >&2
    exit 2
fi

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
release="$("${script_directory}/sync.sh")"
remote_host="${PSYDUCK_REMOTE_HOST:?PSYDUCK_REMOTE_HOST must name an SSH host or user@host}"

ssh_options=(
    -o BatchMode=yes
    -o PreferredAuthentications=publickey
    -o PasswordAuthentication=no
    -o KbdInteractiveAuthentication=no
)

printf -v quoted_release '%q' "${release}"
remote_command="cd ${quoted_release} && exec deployment/remote/execute.sh"

for argument in "$@"; do
    printf -v quoted_argument '%q' "${argument}"
    remote_command+=" ${quoted_argument}"
done

ssh "${ssh_options[@]}" "${remote_host}" "${remote_command}"
