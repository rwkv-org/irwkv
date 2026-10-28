#!/bin/sh

if [ -z "${_IRWKV_INTERNAL_NUMACTL_ARGS:-}" ]; then
    echo "_IRWKV_INTERNAL_NUMACTL_ARGS is not set" >&2
    exit 1
fi

if [ -z "${_IRWKV_INTERNAL_NUMACTL_PYTHON_EXECUTABLE:-}" ]; then
    echo "_IRWKV_INTERNAL_NUMACTL_PYTHON_EXECUTABLE is not set" >&2
    exit 1
fi

if ! command -v numactl >/dev/null 2>&1; then
    echo "numactl is not available on PATH" >&2
    exit 1
fi

case "${_IRWKV_INTERNAL_NUMACTL_ARGS}" in
    *[![:alnum:]\ _=,./-]*)
        echo "Invalid characters in _IRWKV_INTERNAL_NUMACTL_ARGS" >&2
        exit 1
        ;;
esac

# shellcheck disable=SC2086  # word splitting is intentional: this expands into multiple numactl flags
exec numactl ${_IRWKV_INTERNAL_NUMACTL_ARGS} "${_IRWKV_INTERNAL_NUMACTL_PYTHON_EXECUTABLE}" "$@"
