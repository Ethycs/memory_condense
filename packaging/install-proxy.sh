#!/usr/bin/env bash
# Linux x86_64 counterpart of install-proxy.ps1. Keep the two in step: the
# generated Pixi manifest differs only in its platforms line, which is what
# makes the "different manifest" refusal below stable across releases.
#
# bash rather than POSIX sh on purpose: /bin/sh is dash on Debian and Ubuntu,
# which has no pipefail and no arrays, and the package glob needs both.
set -euo pipefail

fail() { printf '%s\n' "$*" >&2; exit 1; }

usage() {
    cat >&2 <<'USAGE'
Usage: install-proxy.sh [options]
  --package-path PATH   The .conda produced by pixi build --output-dir dist
  --install-dir DIR     Application directory (default:
                        ${XDG_DATA_HOME:-$HOME/.local/share}/memory_condense/proxy)
  --assets-dir DIR      Verify and reuse an existing model cache instead of downloading
  --skip-setup          Install the environment only; skip model setup and diagnostics
  -h, --help            Show this message
USAGE
    exit "${1:-2}"
}

package_path=''
install_dir=''
assets_dir=''
skip_setup=0
while [ $# -gt 0 ]; do
    case "$1" in
        --package-path=*) package_path=${1#*=} ;;
        --package-path)   package_path=${2:-}; shift ;;
        --install-dir=*)  install_dir=${1#*=} ;;
        --install-dir)    install_dir=${2:-}; shift ;;
        --assets-dir=*)   assets_dir=${1#*=} ;;
        --assets-dir)     assets_dir=${2:-}; shift ;;
        --skip-setup)     skip_setup=1 ;;
        -h|--help)        usage 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage ;;
    esac
    shift
done

[ -n "${HOME:-}" ] || [ -n "$install_dir" ] || fail 'HOME is unset; pass --install-dir explicitly.'
: "${install_dir:=${XDG_DATA_HOME:-$HOME/.local/share}/memory_condense/proxy}"

command -v pixi >/dev/null 2>&1 || fail 'Install Pixi from https://pixi.sh/ first, then rerun this installer.
If Pixi is installed, add its bin directory to PATH (usually ~/.pixi/bin).'

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)
if [ -z "$package_path" ]; then
    shopt -s nullglob
    candidates=("$script_dir"/memory_condense-*.conda)
    shopt -u nullglob
    [ "${#candidates[@]}" -eq 1 ] \
        || fail 'Pass --package-path with the .conda file produced by pixi build --output-dir dist.'
    package_path=${candidates[0]}
fi
[ -f "$package_path" ] || fail "PackagePath is not a file: $package_path"
case "$package_path" in
    *.conda) ;;
    *) fail '--package-path must be a .conda package.' ;;
esac

# Lowercase hex only: Pixi rejects an uppercase sha256 in a url dependency,
# which is why the PowerShell installer calls ToLowerInvariant.
file_sha256() {
    local out
    if command -v sha256sum >/dev/null 2>&1; then out=$(sha256sum -- "$1")
    elif command -v shasum >/dev/null 2>&1; then out=$(shasum -a 256 -- "$1")
    elif command -v openssl >/dev/null 2>&1; then out=$(openssl dgst -sha256 -r -- "$1")
    else fail 'No sha256 tool found; install coreutils (sha256sum).'
    fi
    printf '%s' "${out%% *}" | tr 'A-F' 'a-f'
}

# A raw space or '#' in the install path would break the Pixi url dependency,
# so percent-encode everything outside the unreserved set. PowerShell gets this
# for free from Uri.AbsoluteUri.
percent_encode_path() {
    local string=$1 index char out=''
    for (( index = 0; index < ${#string}; index++ )); do
        char=${string:index:1}
        case "$char" in
            [A-Za-z0-9/._~-]) out+=$char ;;
            *) out+=$(LC_ALL=C printf '%%%02X' "'$char") ;;
        esac
    done
    printf '%s' "$out"
}

mkdir -p -- "$install_dir"
destination=$(CDPATH='' cd -- "$install_dir" && pwd -P)
installed_package="$destination/$(basename -- "$package_path")"
source_package=$(CDPATH='' cd -- "$(dirname -- "$package_path")" && pwd -P)/$(basename -- "$package_path")
if [ "$source_package" != "$installed_package" ]; then
    if [ -e "$installed_package" ]; then
        [ "$(file_sha256 "$installed_package")" = "$(file_sha256 "$source_package")" ] \
            || fail 'A different package already exists here; select a new --install-dir.'
    else
        install -m 0644 -- "$source_package" "$installed_package"
    fi
fi

package_url="file://$(percent_encode_path "$installed_package")"
package_sha=$(file_sha256 "$installed_package")
manifest_path="$destination/pixi.toml"
manifest=$(cat <<EOF
[workspace]
name = "memory-condense-proxy"
channels = ["conda-forge"]
platforms = ["linux-64"]

[system-requirements]
cuda = "12.0"

[activation.env]
KMP_DUPLICATE_LIB_OK = "TRUE"

[dependencies]
memory_condense = { url = "$package_url", sha256 = "$package_sha" }

[pypi-dependencies]
fastembed = "==0.8.1"
onnxruntime = "==1.30.0"
nvidia-nvcomp-cu12 = "==5.3.0.16"

[tasks]
proxy = "memory-condense proxy"
doctor = "memory-condense doctor"
EOF
)

if [ -e "$manifest_path" ]; then
    # Command substitution strips trailing newlines on both sides, which
    # reproduces the PowerShell installer's .Trim() comparison.
    [ "$(cat -- "$manifest_path")" = "$manifest" ] \
        || fail '--install-dir already contains a different Pixi manifest; select a new directory.'
else
    printf '%s\n' "$manifest" >"$manifest_path"
fi

pixi install --manifest-path "$manifest_path" || fail 'Pixi dependency installation failed.'

launcher="$destination/memory-condense"
cat >"$launcher" <<'LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail
# readlink -f so a symlink into ~/.local/bin still finds the manifest, and exec
# so Ctrl+C reaches uvicorn directly; stopping cleanly is what drains ingestion.
here=$(dirname -- "$(readlink -f -- "$0")")
exec pixi run --manifest-path "$here/pixi.toml" memory-condense "$@"
LAUNCHER
chmod 0755 -- "$launcher"

if [ "$skip_setup" -eq 0 ]; then
    if [ -n "$assets_dir" ]; then
        pixi run --manifest-path "$manifest_path" memory-condense setup --reuse "$assets_dir" \
            || fail 'Model asset setup failed; the application environment is installed.'
    else
        pixi run --manifest-path "$manifest_path" memory-condense setup \
            || fail 'Model asset setup failed; the application environment is installed.'
    fi
    pixi run --manifest-path "$manifest_path" memory-condense doctor \
        || fail 'Runtime diagnostics failed; inspect the doctor output above.'
fi

printf 'Installed. Run: %s proxy --openai-base-url https://your-gateway.example/v1\n' "$launcher"
printf 'Optional: ln -s %s ~/.local/bin/memory-condense\n' "$launcher"
