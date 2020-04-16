#!/usr/bin/env bash

format="$1"
base_url="$2"

set -o errexit
set -o nounset


if [[ "$format" == "" ]]; then
    echo "Please specify a download format! Usage: $0 <csv|json> [base_url]" >&2
    exit 126
fi
if [[ "$format" != "csv" ]] && [[ "$format" != "json" ]]; then
    echo "E: Invalid download format '$format'! Usage: $0 <csv|json> [base_url]" >&2
    exit 126
fi
if [[ "$base_url" == "" ]]; then
    base_url="http://localhost:6006/"
fi
if [[ "$base_url" != *"/" ]]; then
    echo "Base URL must end with a slash! Usage: $0 <csv|json> [base_url]" >&2
    exit 126
fi


tags="$(curl -s "$base_url""data/plugin/scalars/tags")"
for run in $(jq -r 'keys[]' <<<"$tags"); do
    echo
    echo "Download scalars of run $run."
    for metric in $(jq -r ".\"$run\" | keys[]" <<<"$tags"); do
        echo "Download scalar $metric of run $run."
        out_file="$run-$metric.$format"
        curl -s "$base_url""data/plugin/scalars/scalars?tag=$metric&run=$run&format=$format" >"$out_file"
        if [[ "$(cat "$out_file")" == "Not found" ]]; then
            echo "Failed to download metric $metric of run $run: Not found." >&2
            rm "$out_file"
            exit 1
        fi
    done
done
