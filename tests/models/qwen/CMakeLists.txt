function(add_qwen_cpu_test target source labels)
  add_executable(${target} ${PROJECT_SOURCE_DIR}/${source})
  target_link_libraries(${target} PRIVATE gufo_core)
  add_test(NAME ${target} COMMAND ${target})
  set_tests_properties(${target} PROPERTIES LABELS "${labels}")
endfunction()

add_qwen_cpu_test(qwen_embedding_rows_test
  tests/models/qwen/embedding_rows_test.cpp
  "cpu;models;qwen;embedding;ssd")
add_test(NAME qwen_embedding_rows_wiring_test
  COMMAND ${Python3_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/tests/models/qwen/embedding_rows_wiring_test.py)
set_tests_properties(qwen_embedding_rows_wiring_test PROPERTIES
  LABELS "cpu;models;qwen;embedding;ssd;source-contract")

# Qwen tests mirror src/models/qwen so focused module, policy, and runtime
# changes have an obvious validation target while preserving CTest names.
add_qwen_cpu_test(qwen_vision_cache_test
  tests/models/qwen/vision_cache_test.cpp
  "cpu;vision;qwen;correctness")
add_qwen_cpu_test(qwen_tokenizer_test
  tests/models/qwen/tokenization/tokenizer_test.cpp
  "cpu;tokenization;qwen")
add_qwen_cpu_test(qwen_chat_template_test
  tests/models/qwen/tokenization/chat_template_test.cpp
  "cpu;tokenization;chat_template")
target_compile_definitions(qwen_chat_template_test PRIVATE
  GUFO_QWEN38_CHAT_TEMPLATE_REFERENCE="${PROJECT_SOURCE_DIR}/src/models/qwen/reference/chat_template.jinja"
  GUFO_CHAT_TEMPLATE_HF_GOLDENS="${PROJECT_SOURCE_DIR}/tests/fixtures/chat_template_hf_goldens.json")

add_qwen_cpu_test(qwen_cpu_oracles_test
  tests/models/qwen/cpu/oracles_test.cpp
  "cpu;models;qwen;oracles")
add_qwen_cpu_test(qwen_forward_test
  tests/models/qwen/cpu/forward_test.cpp
  "cpu;models;qwen;forward;reference")
add_qwen_cpu_test(qwen_generator_test
  tests/models/qwen/cpu/generator_test.cpp
  "cpu;models;qwen;generator")
add_qwen_cpu_test(qwen_ssm_test
  tests/models/qwen/cpu/ssm_test.cpp
  "cpu;models;qwen;ssm;reference")
add_qwen_cpu_test(qwen_oracle_test
  tests/models/qwen/cpu/model_oracle_test.cpp
  "cpu;models;qwen;oracles")

add_qwen_cpu_test(qwen_l1_norm_module_test
  tests/models/qwen/modules/norm_test.cpp
  "cpu;models;qwen;module;norm")
add_qwen_cpu_test(qwen_l1_ffn_module_test
  tests/models/qwen/modules/ffn_test.cpp
  "cpu;models;qwen;module;ffn")
add_qwen_cpu_test(qwen_l1_ssm_module_test
  tests/models/qwen/modules/ssm_test.cpp
  "cpu;models;qwen;module;ssm")

add_qwen_cpu_test(qwen_fusion_route_test
  tests/models/qwen/hip/execution_policy_test.cpp
  "cpu;models;qwen;hip;policy;fusion")
add_qwen_cpu_test(qwen_attention_policy_test
  tests/models/qwen/hip/attention_policy_test.cpp
  "cpu;models;qwen;hip;attention;dispatch")
add_qwen_cpu_test(qwen_gemm_route_test
  tests/models/qwen/gemm_route_test.cpp
  "cpu;models;qwen;gemm;dispatch")

add_qwen_cpu_test(qwen_sn_weight_plan_test
  tests/models/qwen/sn_weight_plan_test.cpp
  "cpu;models;qwen;quant;sn-loader")
add_qwen_cpu_test(qwen_sn_iq3_pair_plan_test
  tests/models/qwen/sn_iq3_pair_plan_test.cpp
  "cpu;models;qwen;quant;sn-loader;iq3")
add_test(NAME qwen_dot4_scratch_capture_guard
  COMMAND ${Python3_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/tests/models/qwen/dot4_scratch_capture_guard.py)
set_tests_properties(qwen_dot4_scratch_capture_guard PROPERTIES
  LABELS "cpu;models;qwen;quant;iq3;graph;source-contract")
add_test(NAME qwen_iq3_prefill_unpack_guard
  COMMAND ${Python3_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/tests/models/qwen/iq3_prefill_unpack_guard.py)
set_tests_properties(qwen_iq3_prefill_unpack_guard PROPERTIES
  LABELS "cpu;models;qwen;quant;iq3;source-contract")
# B1 FFN routing, C8 CPU refusal, C9 profile/fence.
add_qwen_cpu_test(qwen_iq3_serve_profile_test
  tests/models/qwen/iq3_serve_profile_test.cpp
  "cpu;models;qwen;quant;iq3;sn-loader")
add_test(NAME qwen_iq3_serve_fence_guard
  COMMAND ${Python3_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/tests/models/qwen/iq3_serve_fence_guard.py)
set_tests_properties(qwen_iq3_serve_fence_guard PROPERTIES
  LABELS "cpu;models;qwen;quant;iq3;source-contract")
# NPU B-cache logical key for tagged IQ3 gate/up (host only).
add_qwen_cpu_test(qwen_npu_bcache_key_test
  tests/models/qwen/npu_bcache_key_test.cpp
  "cpu;models;qwen;quant;iq3;npu")

# Host repack round-trip and scalar dequant, CPU only.
add_qwen_cpu_test(qwen_sn_q4k_repack_test
  tests/models/qwen/sn_q4k_repack_test.cpp
  "cpu;models;qwen;hip;quant;sn-q4k")

# Negative control: prints the refusal text of every KV storage so the
# before/after bytes can be diffed. Not an assertion test by design.
add_qwen_cpu_test(qwen_kv_refusal_baseline_probe
  tests/models/qwen/kv_refusal_baseline_probe.cpp
  "cpu;models;qwen;kv;turbo4-kv;negative-control")

# The GUFO_DEV_TURBO4_DECODE dev switch, and the CALL SITE that
# decides which phase it is allowed to relax.
add_qwen_cpu_test(qwen_turbo4_dev_decode_switch_test
  tests/models/qwen/turbo4_dev_decode_switch_test.cpp
  "cpu;models;qwen;kv;turbo4-kv;dev-switch")

# Batched phase refusal and the verify-only decode switch.
add_qwen_cpu_test(qwen_e6_batched_phase_refusal_test
  tests/models/qwen/e6_batched_phase_refusal_test.cpp
  "cpu;models;qwen;kv;turbo4-kv;batched-phase-refusal")

# Route telemetry names each distinct executor entry.
add_qwen_cpu_test(qwen_e6_route_mode_source_test
  tests/models/qwen/e6_route_mode_source_test.cpp
  "cpu;models;qwen;turbo4-kv;route-mode")
set_tests_properties(qwen_e6_route_mode_source_test PROPERTIES
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# CPU-tested suffix lookup and head-fusion policy.
add_qwen_cpu_test(qwen_opt73_lookup_drafting_test
  tests/models/qwen/lookup_drafting_test.cpp
  "cpu;models;qwen;speculative;lookup-drafting;lookup")

# FR-Spec draft vocabulary cut: spec parsing and the row-run gather.
add_qwen_cpu_test(qwen_draft_vocab_test
  tests/models/qwen/draft_vocab_test.cpp
  "cpu;models;qwen;speculative;dflash;draft-vocab")

# CPU proof for capture-window arithmetic and row slicing.
add_qwen_cpu_test(qwen_opt68_capture_window_test
  tests/models/qwen/opt68_capture_window_test.cpp
  "cpu;models;qwen;speculative;dflash-drafter;capture-window")

# The per-step dump. Host code, so the JSONL bytes are assertable.
add_qwen_cpu_test(qwen_dev_step_dump_test
  tests/models/qwen/dev_step_dump_test.cpp
  "cpu;models;qwen;kv;turbo4-kv;step-dump")
set_tests_properties(qwen_dev_step_dump_test PROPERTIES
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# The compact-snapshot row stride. CPU only -- the arithmetic and
# the call site, neither of which needs a device buffer.
add_qwen_cpu_test(qwen_b_r4_compact_snapshot_pitch_test
  tests/models/qwen/b_r4_compact_snapshot_pitch_test.cpp
  "cpu;models;qwen;kv;turbo4-kv;compact-snapshot")
set_tests_properties(qwen_b_r4_compact_snapshot_pitch_test PROPERTIES
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# The snapshot copy must move every valid byte. CPU only.
add_qwen_cpu_test(qwen_b_r5_snapshot_copy_bytes_test
  tests/models/qwen/b_r5_snapshot_copy_bytes_test.cpp
  "cpu;models;qwen;kv;turbo4-kv;snapshot-copy")
set_tests_properties(qwen_b_r5_snapshot_copy_bytes_test PROPERTIES
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# GUFO_DEV_TURBO4_KV selects a Turbo4 KV policy at every construction
# path. CPU only -- pure policy resolution, no engine.
add_qwen_cpu_test(qwen_c1c_turbo4_kv_selector_test
  tests/models/qwen/c1c_turbo4_kv_selector_test.cpp
  "cpu;models;qwen;kv;turbo4-kv;turbo4-selector")
set_tests_properties(qwen_c1c_turbo4_kv_selector_test PROPERTIES
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# ATTN6: the attn6 verify reader's packed Turbo6 code offsets == CodeAt, and
# its split plan fits the WMMA reader's partial buffer. Host-only arithmetic.
add_qwen_cpu_test(qwen_attn6_codes_test
  tests/models/qwen/attn6_codes_test.cpp
  "cpu;models;qwen;turbo4;attn6")

# VRD6: the vrd6 verify reader's Turbo6 pair table == CodeAt + CentroidValue
# for every value, and its P / V^T key order is a bijection. Host-only.
add_qwen_cpu_test(qwen_vrd6_codes_test
  tests/models/qwen/vrd6_codes_test.cpp
  "cpu;models;qwen;turbo4;vrd6")

# Per-token prompt feeding for the Turbo4 decode-only policy.
add_qwen_cpu_test(qwen_c1d_turbo4_prompt_feeding_test
  tests/models/qwen/c1d_turbo4_prompt_feeding_test.cpp
  "cpu;models;qwen;kv;turbo4-kv;prompt-feeding")
set_tests_properties(qwen_c1d_turbo4_prompt_feeding_test PROPERTIES
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# Turbo4 KV unit tests, first real execution. Every one of these is host
# code: the headers they include declare no HIP runtime, and they launch no
# kernel. They were written and code-reviewed but never registered, so none of
# them had ever run.
foreach(turbo4_ab_test
    arena_edit_boundary_test
    kv_allocation_turbo4_test
    kv_format_table_test
    kv_read_test
    kv_snapshot_layer_copy_test
    kv_snapshot_roundtrip_test
    kv_snapshot_turbo4_test
    kv_turbo4_plane_layout_test
    kv_write_test
    parent_probe_fixes_test
    turbo4_attention_ref_test
    turbo4_batched_kv_write_test
    turbo4_batch_causal_test
    turbo4_codec_test
    turbo4_decode_adapter_test
    turbo4_dequant_f16_test
    turbo4_kernel_ref_test
    turbo4_prefill_ref_test
    turbo4_rope_test
    turbo4_vertical_slice_test
    turbo4_wmma_conformance_test
    turbo56_codec_test
    turbo56_plumbing_test
    turbo56_mixed_test)
  add_qwen_cpu_test(qwen_turbo4_ab_${turbo4_ab_test}
    tests/models/qwen/${turbo4_ab_test}.cpp
    "cpu;models;qwen;kv;turbo4-kv;turbo4")
  # These tests assert against the SOURCE text ("src/models/qwen/hip/arena.cpp"),
  # not just against linked code. CTest runs a test from its build directory, so
  # a source-root-relative read resolves to nothing and every content check fails
  # on an empty string. Running them from the project root is what their own
  # manual compile lines in the file headers assume.
  set_tests_properties(qwen_turbo4_ab_${turbo4_ab_test} PROPERTIES
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endforeach()

add_test(NAME qwen_turbo56_mixed_negative_control
  COMMAND $<TARGET_FILE:qwen_turbo4_ab_turbo56_mixed_test> --negative-control)
set_tests_properties(qwen_turbo56_mixed_negative_control PROPERTIES
  PASS_REGULAR_EXPRESSION "FAIL: each plane uses its own codec and physical slot pitch"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  LABELS "cpu;qwen;turbo56;negative-control")

# E2(b), the ESTIMATE half. Registered outside the loop above because it does not
# assert against source text -- it calls QwenGpuArena::EstimateMemoryUsage and
# measures the scratch as a residual against the fp32 estimate. The companion
# gpu test below asserts the allocation itself, which cannot be checked here.
add_qwen_cpu_test(qwen_turbo4_scratch_allocation_test
  tests/models/qwen/turbo4_scratch_allocation_test.cpp
  "cpu;models;qwen;kv;turbo4-kv;turbo4")

# E2(c), the layer-offset caution. Reads the three attention routes' source, so
# it needs the project root as its working directory, like the ab tests above.
# It also touches real device memory for the layer-1 address demonstration; on a
# CPU-only box that part skips itself and the source and arithmetic parts still
# run.
add_qwen_cpu_test(qwen_turbo4_layer_offset_test
  tests/models/qwen/turbo4_layer_offset_test.cpp
  "cpu;models;qwen;kv;turbo4-kv;turbo4")
set_tests_properties(qwen_turbo4_layer_offset_test PROPERTIES
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# E3 rule 1: the ONE helper that builds a block layer's f16 pair. It has no
# layer parameter, so K, V and the layer index cannot drift apart at a call
# site. The test reads the header's own signature, so a parameter added later
# fails here rather than compiling everywhere.
add_qwen_cpu_test(qwen_turbo4_attention_pair_test
  tests/models/qwen/turbo4_attention_pair_test.cpp
  "cpu;models;qwen;kv;turbo4-kv;turbo4")
set_tests_properties(qwen_turbo4_attention_pair_test PROPERTIES
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# E3 rules 5 and 6: the switch-unset path takes every f16 arm, and a block layer
# stays graph-ineligible. Checks the DECISIONS the f16 bytes depend on plus the
# wiring shape -- a full byte-identical run needs the model on a GPU and is E4.
add_qwen_cpu_test(qwen_turbo4_prefill_wiring_test
  tests/models/qwen/turbo4_prefill_wiring_test.cpp
  "cpu;models;qwen;kv;turbo4-kv;turbo4")
set_tests_properties(qwen_turbo4_prefill_wiring_test PROPERTIES
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

# The row-interleaved IQ3_S repack (sn_iq3.hpp). Byte-exact round
# trip, the scalar dequant read straight out of the repacked planes, and the
# +2% size contract. Host only -- no HIP, no GPU.
add_qwen_cpu_test(qwen_sn_iq3_repack_test
  tests/models/qwen/sn_iq3_repack_test.cpp
  "cpu;models;qwen;quant;iq3;repack;sn-iq3")

# The LDS-address decode of OpusIq3sB8CutKernel (sn_iq3_cut.hpp)
# reads the same table entry as the definition, for all three index modes.
add_qwen_cpu_test(qwen_sn_iq3_cut_test
  tests/models/qwen/sn_iq3_cut_test.cpp
  "cpu;models;qwen;quant;iq3;sn-iq3")

if (ENGINE_ENABLE_HIP)
  function(add_qwen_hip_test target source labels)
    add_executable(${target} ${PROJECT_SOURCE_DIR}/${source})
    target_link_libraries(${target} PRIVATE
      gufo_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES
      LABELS "${labels}"
      SKIP_RETURN_CODE 77)
  endfunction()

  # GPU IQ3 unpack byte-equal to native (P, G, U).
  add_qwen_hip_test(qwen_iq3_unpack_gpu_test
    tests/models/qwen/iq3_unpack_gpu_test.cpp
    "hip;gpu;qwen;quant;iq3;sn-iq3;slow")

  # BOTH GPU TESTS WERE UNREGISTERED until now: each built only
  # from a hand-written hipcc line in its own header comment, so a fresh
  # checkout and a plain `ctest` never ran them and a green suite never covered
  # either. Registered here with the `gpu` label and SKIP_RETURN_CODE 77 above,
  # so a CPU-only ctest skips them instead of failing.
  add_qwen_hip_test(qwen_turbo4_batched_kv_write_gpu_test
    tests/models/qwen/turbo4_batched_kv_write_gpu_test.cpp
    "hip;gpu;qwen;turbo4;kv;turbo4-kv;slow")

  # E2 dequant: pins the #pragma clang fp contract(off) at the top of
  # kernels/turbo4_kv_dequant.hip. Without contraction off this reports ~0.17%
  # of elements differing from the host body, 1 ULP apart, spread over many
  # positions. If that pragma is ever dropped or moved after the includes,
  # THIS TEST is what catches it.
  add_qwen_hip_test(qwen_turbo4_dequant_f16_gpu_test
    tests/models/qwen/turbo4_dequant_f16_gpu_test.cpp
    "hip;gpu;qwen;turbo4;dequant;turbo4-kv;slow")

  # E2(b) allocation half. The CPU test above proves the ESTIMATE charges the
  # scratch; this proves the allocator performs the hipMalloc, records the
  # number it actually asked for, and that the bytes behind the pointer are
  # real. The check that only exists here is the cross-site one: allocated
  # bytes == estimated bytes.
  add_qwen_hip_test(qwen_turbo4_scratch_alloc_gpu_test
    tests/models/qwen/turbo4_scratch_alloc_gpu_test.cpp
    "hip;gpu;qwen;turbo4;kv;turbo4-kv;slow")

  add_qwen_hip_test(qwen_turbo4_attention_split_gpu_test
    tests/models/qwen/turbo4_attention_split_gpu_test.cpp
    "hip;gpu;qwen;turbo4;decode;turbo4-kv;slow")

  # Batched multi-row verify reader, bit-identical to the per-row one.
  add_qwen_hip_test(qwen_turbo4_attention_batched_gpu_test
    tests/models/qwen/turbo4_attention_batched_gpu_test.cpp
    "hip;gpu;qwen;turbo4;decode;verify-reader;slow")

  # WMMA verify reader vs the scalar split reader (L2 rel < 2e-3, not
  # bit-identical). Run with GUFO_DEV_TURBO4_VERIFY_WMMA unset; the test calls
  # the launcher directly.
  add_qwen_hip_test(qwen_turbo4_verify_wmma_gpu_test
    tests/models/qwen/turbo4_verify_wmma_gpu_test.cpp
    "hip;gpu;qwen;turbo4;decode;verify-reader;slow")

  # Turbo5/Turbo6 writer, dequant and both readers on device,
  # with Turbo4 as the control. Authored without a GPU run; run before use.
  add_qwen_hip_test(qwen_turbo56_gpu_test
    tests/models/qwen/turbo56_gpu_test.cpp
    "hip;gpu;qwen;turbo4;turbo56;kv;slow")

  add_qwen_hip_test(qwen_hip_device_gate_test
    tests/models/qwen/hip/device_gate_test.cpp
    "hip;qwen;test-support;device-gate")

  add_qwen_hip_test(qwen_gpu_ops_test
    tests/models/qwen/hip/basic/blas_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;basic;blas")
  # LaunchGEMVRows bitwise equal to the per-row LaunchGEMV loop.
  add_qwen_hip_test(qwen_gemv_rows_gpu_test
    tests/models/qwen/hip/basic/gemv_rows_gpu_test.cpp
    "hip;gpu;kernels;qwen;gemv-rows;basic;blas")
  add_qwen_hip_test(qwen_elementwise_ops_test
    tests/models/qwen/hip/basic/elementwise_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;basic;elementwise")

  add_qwen_hip_test(qwen_attention_decode_ops_test
    tests/models/qwen/hip/attention/backend_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;attention;decode;backend")
  add_qwen_hip_test(qwen_attention_long_context_ops_test
    tests/models/qwen/hip/attention/long_context_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;attention;decode;long-context")
  add_qwen_hip_test(qwen_attention_kv_storage_ops_test
    tests/models/qwen/hip/attention/kv_storage_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;attention;decode;long-context;snapshot;opt-c0133-fp16-kv")
  add_qwen_hip_test(qwen_attention_fusion_ops_test
    tests/models/qwen/hip/attention/qk_rope_kv_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;attention;fusion;opt-c010-qk-rope-kv")
  add_qwen_hip_test(qwen_attention_wmma_prefill_ops_test
    tests/models/qwen/hip/attention/wmma_prefill_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;attention;prefill;opt-c177-attn-wmma")
  add_qwen_hip_test(qwen_attention_projection_ops_test
    tests/models/qwen/hip/attention/projection_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;attention;ssm;projection")
  add_qwen_hip_test(qwen_attention_component_ops_test
    tests/models/qwen/hip/attention/component_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;attention;component")

  add_qwen_hip_test(qwen_ssm_ops_test
    tests/models/qwen/hip/ssm_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;ssm;opt-c0138-bf16-recurrent")

  add_qwen_hip_test(qwen_graph_ops_test
    tests/models/qwen/hip/graph_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;graph")
  add_qwen_hip_test(qwen_quant_gemv_ops_test
    tests/models/qwen/hip/quant/q8_gemv_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;quant;gemv;q8;opt-c162-q8k-model;opt-c206-q8-small-batch")
  add_qwen_hip_test(qwen_prefill_quant_gemm_ops_test
    tests/models/qwen/hip/quant/prefill_quant_gemm_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;quant;gemm;prefill;q8;opt-c163-blocked-w8a8;opt-c192-swiglu-epilogue")
  add_qwen_hip_test(qwen_q4kxl_quant_ops_test
    tests/models/qwen/hip/quant/q4kxl_quant_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;quant;gemv;gemm;k-quant;iquant;opt-q4kxl")
  # Bit-exact device decode of the extra iron quants. Build only; the
  # GPU run is a separate step.
  add_qwen_hip_test(qwen_iquant_decode_gpu_test
    tests/models/qwen/hip/quant/iquant_decode_gpu_test.hip
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;quant;iquant;extra-quants")
  add_qwen_hip_test(qwen_kquant_gemv_ops_test
    tests/models/qwen/hip/quant/kquant_gemv_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;quant;gemv;k-quant")
  add_qwen_hip_test(qwen_dequant_ops_test
    tests/models/qwen/hip/dequant_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;quant;dequant;opt-c162-q8k-model")
  add_qwen_hip_test(qwen_module_ops_test
    tests/models/qwen/hip/module_ops_test.cpp
    "hip;gpu;kernels;qwen;qwen-kernel-oracle;slow;module")
  add_qwen_hip_test(qwen_l2_gpu_integration_test
    tests/models/qwen/hip/integration_test.cpp
    "hip;gpu;kernels;qwen;module;l2")
endif()
