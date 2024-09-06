#!/usr/bin/env bash

set -e

QDRANT_PATH=$1

QDRANT_PATH=$(realpath $QDRANT_PATH)

# Get path to this script
SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
ROOT_PATH=$SCRIPT_PATH/..

# Extract all files from QDRANT_PATH with .rs extension and write to json file.
python -m code_search.index.files_to_json

# Read the json file and upload it to the Qdrant collection named "code-files".
# Note that it just contains payload without any vectors. Example payload uploaded to collection.
# {
#     "path": "lib\\api\\build.rs",
#     "code": ["use std::path::PathBuf;\n", "use std::process::Command;\n"],
#     "startline": 1,
#     "endline": 352
# }
python -m code_search.index.file_uploader

# We have created a dockerized setup that generates the index.lsif file for us.
# Refer to the docker/rust-analyzer/Dockerfile for more details.
# To generate the index.lsif file, download the required Rust project (In our case, qdrant code base) and update the 
# volumes in docker compose and execute "docker compose up --build"
rustup run stable rust-analyzer -v lsif $QDRANT_PATH > $ROOT_PATH/data/index.lsif

# Convert the index.lsif file to qdrant_snippets.jsonl for the next step.
python -m code_search.index.convert_lsif_index

# Read the qdrant_snippets.jsonl file and upload it to the Qdrant collection named "code-snippets-unixcoder".
python -m code_search.index.upload_code

# Set env variables QDRANT_PATH(Qdrant Code base) and ROOT_PATH(working directory)
# Parse the qdrant source code and generate a structures.jsonl file.
# Example format of the json generated.

# {
#   "name": "search_with_graph",
#   "signature": "fn search_with_graph (& self , vector : & QueryVector , filter : Option < & Filter > , top : usize , params : Option < & SearchParams > , custom_entry_points : Option < & [PointOffsetType] > , vector_query_context : & VectorQueryContext ,) -> OperationResult < Vec < ScoredPointOffset > >",
#   "code_type": "Function",
#   "docstring": null,
#   "line": 498,
#   "line_from": 497,
#   "line_to": 539,
#   "context": {
#     "module": "hnsw_index",
#     "file_path": "lib/segment/src/index/hnsw_index/hnsw.rs",
#     "file_name": "hnsw.rs",
#     "struct_name": "HNSWIndex < TGraphLinks >",
#     "snippet": "    #[allow(clippy::too_many_arguments)]\n    fn search_with_graph(\n        &self,\n        vector: &QueryVector,\n        filter: Option<&Filter>,\n        top: usize,\n        params: Option<&SearchParams>,\n        custom_entry_points: Option<&[PointOffsetType]>,\n        vector_query_context: &VectorQueryContext,\n    ) -> OperationResult<Vec<ScoredPointOffset>> {\n        let ef = params\n            .and_then(|params| params.hnsw_ef)\n            .unwrap_or(self.config.ef);\n\n        let is_stopped = vector_query_context.is_stopped();\n\n        let id_tracker = self.id_tracker.borrow();\n        let payload_index = self.payload_index.borrow();\n        let vector_storage = self.vector_storage.borrow();\n        let quantized_vectors = self.quantized_vectors.borrow();\n\n        let deleted_points = vector_query_context\n            .deleted_points()\n            .unwrap_or(id_tracker.deleted_point_bitslice());\n\n        let raw_scorer = Self::construct_search_scorer(\n            vector,\n            &vector_storage,\n            quantized_vectors.as_ref(),\n            deleted_points,\n            params,\n            &is_stopped,\n        )?;\n        let oversampled_top = Self::get_oversampled_top(quantized_vectors.as_ref(), params, top);\n\n        let filter_context = filter.map(|f| payload_index.filter_context(f));\n        let points_scorer = FilteredScorer::new(raw_scorer.as_ref(), filter_context.as_deref());\n\n        let search_result =\n            self.graph\n                .search(oversampled_top, ef, points_scorer, custom_entry_points);\n        self.postprocess_search_result(search_result, vector, params, top, &is_stopped)\n    }\n"
#   }
# }
docker run --rm -v $QDRANT_PATH:/source qdrant/rust-parser ./rust_parser /source > $ROOT_PATH/data/structures.jsonl

# Read the structures.jsonl file and upload it to the Qdrant collection named "code-signatures".
python -m code_search.index.upload_signatures
