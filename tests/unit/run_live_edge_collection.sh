#!/usr/bin/env bash
set -euo pipefail

edge_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
edge_build_dir="${FALKORDB_EDGE_TEST_BUILD_DIR:-$edge_repo_root/bin/live-edge-tests}"
edge_graphblas_dir="${FALKORDB_EDGE_TEST_GRAPHBLAS_DIR:-$edge_build_dir/graphblas}"
edge_cc="${CC:-cc}"
mkdir -p "$edge_build_dir"

if [[ ! -f "$edge_graphblas_dir/libgraphblas.dylib" && ! -f "$edge_graphblas_dir/libgraphblas.so" ]]; then
    cmake -S "$edge_repo_root/deps/GraphBLAS" -B "$edge_graphblas_dir" \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
        -DGRAPHBLAS_COMPACT=ON -DGRAPHBLAS_USE_JIT=OFF \
        -DGRAPHBLAS_USE_OPENMP=OFF -DGRAPHBLAS_USE_CUDA=OFF
    cmake --build "$edge_graphblas_dir" -j "${FALKORDB_EDGE_TEST_JOBS:-2}"
fi

edge_compile_flags=(-std=gnu11 -O1 -g -ffunction-sections -fdata-sections -DFALKORDB_FOCUSED_EDGE_TEST)
if [[ "${FALKORDB_EDGE_TEST_DEBUG:-0}" == 1 ]]; then
    edge_compile_flags+=(-DRG_DEBUG)
fi
if [[ "${FALKORDB_EDGE_TEST_SANITIZER:-}" == address ]]; then
    edge_compile_flags+=(-fsanitize=address -fno-omit-frame-pointer)
fi
if [[ "$(uname -s)" == Darwin ]]; then
    edge_link_flags=(-Wl,-dead_strip)
else
    edge_link_flags=(-Wl,--gc-sections -lm -pthread)
fi

cd "$edge_repo_root"
edge_delta_sources=()
for edge_source in src/graph/delta_matrix/*.c; do
    case "$edge_source" in
        # These write-only translation units need the generated Cypher parser
        # headers. No function in them is used by this focused collector test.
        */delta_remove_row.c) continue ;;
    esac
    edge_delta_sources+=("$edge_source")
done

"$edge_cc" "${edge_compile_flags[@]}" \
    -I. -Isrc -Ideps -Ideps/rax -Ideps/xxHash -Ideps/GraphBLAS/Include \
    tests/unit/test_live_edge_collection.c \
    src/graph/graph.c src/graph/graph_collect_node_edges.c src/graph/entities/edge.c \
    src/graph/tensor/*.c "${edge_delta_sources[@]}" \
    src/util/datablock/*.c src/util/block.c \
    -L"$edge_graphblas_dir" -lgraphblas "${edge_link_flags[@]}" \
    -Wl,-rpath,"$edge_graphblas_dir" -o "$edge_build_dir/test_live_edge_collection"
"$edge_build_dir/test_live_edge_collection"
