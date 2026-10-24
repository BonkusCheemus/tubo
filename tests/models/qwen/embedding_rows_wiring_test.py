#!/usr/bin/env python3
"""CPU source contracts only; this does NOT qualify HIP execution or speed."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]


def source(path):
    text = (ROOT / 'src/models/qwen' / path).read_text()
    text = re.sub(r'//[^\n]*|/\*.*?\*/', '', text, flags=re.S)
    return re.sub(r'\s+', '', text)


decode = source('hip/decode.cpp')
assert decode.index('decode_embedding_.Prepare(') < decode.index('graph_executor_.TryCapture(')
assert decode.count('ExecuteDecodeStep(') == 3
assert decode.count('&vision_input_,decode_embedding_.Data())') == 3
step = source('hip/decode_step.cpp')
assert 'if(embedding_row){LaunchEmbeddingLookup(embedding_row,weights.token_embd.type,0U,' in step
assert 'else{LaunchEmbeddingLookup(weights.token_embd.data,' in step
header = source('hip/executor.hpp')
assert 'EmbeddingRowStagedecode_embedding_;EmbeddingRowStagebatch_embedding_;' in header
loader = source('hip/model_loader.cpp')
assert loader.index('EmbeddingRowStore::Open(') < loader.index('CreateWeightRegions(*reader,')
assert '(embedding_rows||remap(weights_opt->token_embd))' in loader
assert 'EmbeddingResidentSpans(' in loader
assert 'MapRegisteredRegion(slice,destination,span.offset)' in loader
assert 'CopyRegionToDevice(source,destination,file_offset)' in loader
assert 'CopyMappedWeights(source,host_copy,file_offset)' in loader
assert loader.count('static_cast<off_t>(file_offset+offset)') == 2
assert 'if(embedding_rows&&embedding_rows->Shard()==i)' in loader
assert 'MADV_DONTNEED' in loader and 'POSIX_FADV_DONTNEED' in loader
prefill = source('hip/prefill_chunk.cpp')
assert 'batch_embedding_.LaunchBatch(model_->GetEmbeddingRows(),weights_.token_embd,host_tokens,' in prefill
assert 'hidden_size,arena_.stream,true)' in prefill
assert source('hip/batched_decode.cpp').count('batch_embedding_.LaunchBatch(') == 2
assert 'embedding_stage_.LaunchSingle(model_->GetEmbeddingRows(),weights.token_embedding,' in source('hip/mtp/executor.cpp')
assert 'embedding_stage_.LaunchBatch(model_->GetEmbeddingRows(),weights.token_embedding,block_tokens,' in source('hip/dflash/executor.cpp')
assert 'coordinator.embedding_stage_.LaunchBatch(coordinator.model_->GetEmbeddingRows(),weights.token_embedding,tokens,' in source('hip/dflash/batch.cpp')
for model in ('hip/mtp.hpp', 'hip/dflash.hpp'):
    assert 'returntarget_model_->GetEmbeddingRows();' in source(model)
stage = source('hip/embedding_rows.cpp')
assert stage.index('hipEventSynchronize(copied_)') < stage.index('store.Fetch(')
assert stage.index('store.Fetch(') < stage.index('hipMemcpyAsync(device_rows_')
assert 'EmbeddingStageCapacity(ids.size(),capacity_rows_,store.RowBytes())' in stage
assert 'capacity_bytes_=allocation_bytes;capacity_rows_=capacity;' in stage
assert 'hipMalloc(&device_rows_,allocation_bytes)' in stage
assert 'capacity*sizeof(std::uint32_t)' in stage
assert 'LaunchBatchedEmbeddingLookup(device_rows_,table.type,device_ids_,' in stage
assert 'LaunchBatchedEmbeddingLookup(table.data,table.type,device_ids,' in stage
store = source('embedding_rows.cpp')
assert 'kPrefillWorkers=8;' in store
assert 'start.count_down();' in store and 'workers.emplace_back(' in store
assert 'while(!stopped.load(std::memory_order_acquire))' in store
assert 'index>=unique.size()||stopped.load(std::memory_order_acquire)' in store
assert store.count('stopped.store(true,std::memory_order_release)') == 2
assert 'run();' in store
serving = re.sub(r'\s+', '', (ROOT / 'tests/models/qwen27b/serving_test.cpp').read_text())
assert 'model->GetEmbeddingRows()?region_count>=shard_count&&region_count<=shard_count+1:region_count==shard_count' in serving
assert 'O_RDONLY|O_CLOEXEC|O_DIRECT' in store
print('PASS: CPU source wiring — graph refresh, both loader modes, all target/draft/MTP gathers, encoded staging, direct reads. No device qualification.')
