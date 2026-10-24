// Regression suite for the defects an independent review probe found in the Turbo4 KV code.
//
//   g++ -std=c++20 -Wall -Wextra -I.
//   tests/models/qwen/parent_probe_fixes_test.cpp -o /tmp/pp
//
// Every case here is one case of an independent reviewer's `parent_review_probe.cpp`
// reproduced against my code. Each was a real defect; each now has a check
// that fails on the old behaviour. They are written as the PROBE states them,
// not as my implementation would have phrased them — the value of an
// independent probe is that it does not share my blind spots.
//
//   1. CheckedEncode returned TRUE for a block that stored inf.
//   2. A 256-value head wrote successfully, then read back refused.
//   3. A hostile geometry wrapped first_block to 0 and Read returned TRUE.
//   4. Storage enum 99 reported 4 B/element and read as valid.
//
// All four share one theme: **a refusal that returned success, or a success
// that produced bytes nobody can read back.**

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "src/models/qwen/hip/kv_allocation.hpp"
#include "src/models/qwen/hip/kv_read.hpp"
#include "src/models/qwen/hip/turbo4_codec.hpp"

namespace t4 = gufo::hip::turbo4;
using gufo::hip::QwenKvCacheStorage;
using gufo::hip::QwenKvPlaneGeometry;
using gufo::hip::QwenKvReadHead;
using gufo::hip::QwenKvWriteHead;

namespace {
int g_failures = 0;
void Check(bool ok, const std::string& what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok)
    ++g_failures;
}

/// Build 128 values whose L2 norm is exactly `target`, by placing all the
/// energy in one element. Deterministic and independent of any rounding in a
/// sinusoid.
std::vector<float> BlockWithNorm(float target) {
  std::vector<float> v(t4::kBlockValues, 0.0f);
  v[0] = target;
  return v;
}
}  // namespace

int main() {
  std::printf("Turbo4 KV regression suite for the independent review probe\n");

  // ---- 1. CheckedEncode MUST refuse a block that stores inf --------------
  // probe: raw norm 65480.007812, recon 0.962773, stored = 68011.9 -> inf.
  // The old guard compared the RAW norm against 65504 and passed.
  {
    // Find a raw norm whose corrected value overflows: corrected =
    // raw/recon, and recon is what the encoder actually computes, so probe for
    // it rather than assuming.
    const float raw = 65480.007812f;
    const auto v = BlockWithNorm(raw);
    std::uint8_t blk[t4::kBlockBytes] = {0};
    const bool ok = t4::QuantizeBlockChecked(v.data(), blk);
    std::uint16_t stored = 0;
    if (ok)
      std::memcpy(&stored, blk, sizeof(stored));
    const float stored_f = t4::HalfBitsToFloat(stored);
    std::printf("  note  raw norm %.3f -> stored %.1f (fp16 max %.0f)\n",
                static_cast<double>(raw), static_cast<double>(stored_f),
                static_cast<double>(t4::kNormMaxFinite));
    Check(!ok || t4::GroupNormIsStorable(stored_f),
          "CheckedEncode returns true ONLY when the STORED norm is storable");

    // And the decisive form: any block at all whose written norm overflows
    // must be refused.
    std::size_t refused = 0, accepted = 0, bad_accepted = 0;
    for (float target = 60000.0f; target < 66000.0f; target += 10.0f) {
      const auto probe = BlockWithNorm(target);
      std::uint8_t p[t4::kBlockBytes] = {0};
      const bool got = t4::QuantizeBlockChecked(probe.data(), p);
      std::uint16_t h = 0;
      std::memcpy(&h, p, sizeof(h));
      const float stored_v = t4::HalfBitsToFloat(h);
      if (got) {
        ++accepted;
        if (!t4::GroupNormIsStorable(stored_v))
          ++bad_accepted;
      } else {
        ++refused;
      }
    }
    std::printf(
        "  note  swept norms 60000..66000: %zu refused, %zu accepted, "
        "%zu accepted-but-unstorable\n",
        refused, accepted, bad_accepted);
    Check(bad_accepted == 0,
          "across the whole sweep, NO accepted block stored an unusable norm");
  }

  // ---- 2. WRITE and READ must AGREE --------------------------------------
  // probe: a 256-value head wrote successfully, then the read refused.
  {
    // head_dim 256 = two blocks. Make block 0 fine and block 1 such that the
    // corrected norm overflows, so the head is un-writable.
    std::vector<float> head(256);
    for (std::size_t i = 0; i < 128; ++i)
      head[i] = 1.0f;
    for (std::size_t i = 128; i < 256; ++i)
      head[i] = 65000.0f;

    const QwenKvPlaneGeometry g{256, 1, 1};
    const std::size_t pb = gufo::hip::QwenKvBytesForElements(
        QwenKvCacheStorage::kTurbo4, 256 * 1 * 1);
    std::vector<std::uint8_t> plane(pb, 0);
    const bool wrote = QwenKvWriteHead(QwenKvCacheStorage::kTurbo4,
                                       plane.data(), pb, g, 0, 0, head.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
    std::vector<float> out(256, -1.0f);
    const bool read = QwenKvReadHead(QwenKvCacheStorage::kTurbo4, plane.data(),
                                     pb, g, 0, 0, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
    std::printf("  note  wrote=%s read=%s\n", wrote ? "true" : "false",
                read ? "true" : "false");
    Check(!(wrote && !read),
          "a successful write is ALWAYS followed by a successful read");
    if (!wrote) {
      std::size_t touched = 0;
      for (std::uint8_t b : plane)
        if (b != 0)
          ++touched;
      Check(touched == 0, "a refused write leaves the plane untouched");
    }
  }

  // ---- 2b. REPORTED EXACT CASE: second block, raw 65480.007812 ---------
  // Her probe distinguishes this from the case I originally tested, and the
  // distinction is the whole point:
  //
  //   raw norm 65480.007812  is UNDER the fp16 ceiling 65504, so a guard that
  //   checks the RAW norm PASSES it. But the value actually stored is
  //   group_norm / recon, and with recon = 0.962773 that is 68011.9, which
  //   overflows to inf.
  //
  // So the defect only shows on a block whose raw norm is JUST UNDER the limit
  // and whose reconstruction factor pushes it OVER. A sweep of "obviously too
  // big" values cannot find it — which is exactly what my first mutant missed.
  {
    constexpr float kShape2Raw = 65480.007812f;
    std::vector<float> head(256);
    for (std::size_t i = 0; i < 128; ++i)
      head[i] = 1.0f;
    // Second block: all the energy in one element, so its L2 norm is exactly
    // the target and nothing is lost to rounding.
    const auto block1 = BlockWithNorm(kShape2Raw);
    for (std::size_t i = 0; i < 128; ++i)
      head[128 + i] = block1[i];

    // Prove the premise: the RAW norm passes a raw-norm guard.
    Check(t4::GroupNormIsStorable(kShape2Raw),
          "the raw norm 65480.007812 is UNDER 65504, so a raw check passes it");
    // And that the block is still unencodable, because of what gets STORED.
    std::uint8_t probe[t4::kBlockBytes] = {0};
    Check(!t4::QuantizeBlockChecked(block1.data(), probe),
          "but the corrected norm overflows fp16, so the encoder REFUSES");

    const QwenKvPlaneGeometry g{256, 1, 1};
    const std::size_t pb =
        gufo::hip::QwenKvBytesForElements(QwenKvCacheStorage::kTurbo4, 256);
    std::vector<std::uint8_t> plane(pb, 0);
    const bool wrote = QwenKvWriteHead(QwenKvCacheStorage::kTurbo4,
                                       plane.data(), pb, g, 0, 0, head.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
    std::printf("  note  probe shape2 head: write=%s (want false)\n",
                wrote ? "true" : "false");
    Check(!wrote, "the 256-head write is REFUSED (parent binary gave true)");
    std::size_t touched = 0;
    for (std::uint8_t b : plane)
      if (b != 0)
        ++touched;
    Check(touched == 0,
          "and block 0 was NOT written either: the refusal is all-or-nothing");

    std::vector<float> out(256, -1.0f);
    const bool read = QwenKvReadHead(QwenKvCacheStorage::kTurbo4, plane.data(),
                                     pb, g, 0, 0, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
    Check(!read || touched == 0, "write and read still agree about this head");
  }

  // ---- 3. A HOSTILE GEOMETRY MUST NOT WRAP TO A VALID OFFSET ------------
  // probe: {head_dim 128, kv_heads 2, context SIZE_MAX+2}, token SIZE_MAX+1,
  // a 132 B plane. The unchecked offset wrapped first_block to 0 and Read
  // returned TRUE, answering from the start of the plane.
  {
    // The probe's exact values: context 9223372036854775809 = 2^63 + 1, token
    // 9223372036854775808 = 2^63. My first attempt wrote these as
    // `(size_t)(-1) + 2`, which is UNSIGNED arithmetic and wraps to 1 — a
    // perfectly valid geometry, so the hostile case silently stopped testing
    // anything.
    constexpr std::size_t kHostileContext = 9223372036854775809ULL;  // 2^63 + 1
    constexpr std::size_t kHostileToken = 9223372036854775808ULL;    // 2^63
    const QwenKvPlaneGeometry hostile{128, 2, kHostileContext};
    std::vector<std::uint8_t> tiny(132, 0);
    std::vector<float> head(128, 0.5f), out(128, -1.0f);
    const bool r =
        QwenKvReadHead(QwenKvCacheStorage::kTurbo4, tiny.data(), tiny.size(),
                       hostile, kHostileToken, 0, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
    std::printf("  note  Read with token 2^63 on a 132 B plane: %s\n",
                r ? "true" : "false");
    Check(!r, "a wrapping index does NOT produce a successful read");
    std::size_t touched = 0;
    for (float v : out)
      if (v != -1.0f)
        ++touched;
    Check(touched == 0, "and it writes nothing into the output");

    const bool w =
        QwenKvWriteHead(QwenKvCacheStorage::kTurbo4, tiny.data(), tiny.size(),
                        hostile, kHostileToken, 0, head.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true);
    Check(!w, "the same wrapping index does NOT produce a successful write");
  }

  // ---- 4. AN UNKNOWN STORAGE ENUM MUST FAIL CLOSED -----------------------
  // probe: value 99 reported 4 B/element and read as valid.
  {
    const auto bogus = static_cast<QwenKvCacheStorage>(99);
    Check(!gufo::hip::QwenKvStorageIsKnown(bogus),
          "enum 99 is not a known storage");
    Check(gufo::hip::QwenKvElementBytes(bogus) == 0,
          "enum 99 reports 0 B/element, not 4");
    Check(!gufo::hip::QwenKvHasElementBytes(bogus),
          "enum 99 declares no element size at all");
    const QwenKvPlaneGeometry g{128, 1, 1};
    Check(!gufo::hip::QwenKvPlaneIsReadable(bogus, g),
          "a plane with an unknown storage is not readable");
    Check(gufo::hip::QwenKvWholePlaneBytes(bogus, g) == 0,
          "an unknown storage has no whole-plane size");
    std::vector<std::uint8_t> buf(4096, 0);
    std::vector<float> out(128, -1.0f), head(128, 0.25f);
    Check(!QwenKvReadHead(bogus, buf.data(), buf.size(), g, 0, 0, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(bogus)), true),
          "reading with an unknown storage is refused");
    Check(!QwenKvWriteHead(bogus, buf.data(), buf.size(), g, 0, 0, head.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(bogus)), true),
          "writing with an unknown storage is refused");
    // Every enumerator we DO know still works.
    for (auto known : {QwenKvCacheStorage::kFp32, QwenKvCacheStorage::kFp16,
                       QwenKvCacheStorage::kTurbo4}) {
      Check(gufo::hip::QwenKvStorageIsKnown(known),
            "a real enumerator is still recognised");
    }
  }

  // ---- 5. WHOLE-PLANE BOUNDS ARE ENFORCED BEFORE ADDRESSING -------------
  {
    const QwenKvPlaneGeometry g{128, 4, 8};
    const std::size_t whole =
        gufo::hip::QwenKvWholePlaneBytes(QwenKvCacheStorage::kTurbo4, g);
    Check(whole == 4 * 8 * 66, "4 heads x 8 tokens x 66 B = the whole plane");
    std::vector<std::uint8_t> plane(whole, 0);
    std::vector<float> head(128, 0.3f), out(128, -1.0f);
    Check(
        QwenKvWriteHead(QwenKvCacheStorage::kTurbo4, plane.data(), whole, g, 7,
                        3, head.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
        "the last head of the last token writes into a correctly sized plane");
    // One byte short must be refused for the WHOLE plane, not just the tail.
    std::vector<std::uint8_t> shortbuf(whole - 1, 0);
    Check(!QwenKvReadHead(QwenKvCacheStorage::kTurbo4, shortbuf.data(),
                          shortbuf.size(), g, 0, 0, out.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
          "a plane one byte short is refused even for token 0 head 0");
    Check(!QwenKvWriteHead(QwenKvCacheStorage::kTurbo4, shortbuf.data(),
                           shortbuf.size(), g, 0, 0, head.data(), gufo::hip::turbo4::PlaneFormats(gufo::hip::QwenKvTurboBits(QwenKvCacheStorage::kTurbo4)), true),
          "and a short plane refuses a write even for token 0 head 0");
  }

  // ---- 5b. THE EXECUTOR DECISION IS A TESTED PREDICATE, NOT AN `if` -----
  // arena.cpp is HIP-only, so a refusal inside it could only be checked by
  // grepping its own source. Moving the decision into a constexpr predicate
  // makes it a CPU-tested invariant instead of a comment.
  //
  // SUPERSEDED, and the correction is the point. This block used to assert
  // `allocatable == has-element-bytes` for every storage, on the reasoning that
  // anything allocatable must have a member and a per-element size. Review
  // rejected that: once the block plane is wired, a Turbo4 plane is
  // allocatable while having NO bytes-per-element, and welding the two forces
  // one of them to be wrong the moment the other is right. They answer
  // different questions. The single implication that must hold is element bytes
  // IMPLIES allocatable; the converse is false and Turbo4 is the witness.
  {
    Check(gufo::hip::QwenKvIsAllocatableByExecutor(QwenKvCacheStorage::kFp16),
          "the executor CAN allocate an fp16 plane");
    Check(gufo::hip::QwenKvIsAllocatableByExecutor(QwenKvCacheStorage::kFp32),
          "the executor CAN allocate an fp32 plane");
    Check(gufo::hip::QwenKvIsAllocatableByExecutor(QwenKvCacheStorage::kTurbo4),
          "the executor CAN now allocate a Turbo4 plane — d_kv_turbo4_ exists "
          "and the block pitch comes from one authority");
    Check(!gufo::hip::QwenKvIsAllocatableByExecutor(
              static_cast<QwenKvCacheStorage>(99)),
          "the executor CANNOT allocate for an unknown precision");
    // The one direction that must hold: a bytes-per-element implies a place to
    // put it.
    for (auto st : {QwenKvCacheStorage::kFp16, QwenKvCacheStorage::kFp32,
                    QwenKvCacheStorage::kTurbo4}) {
      if (gufo::hip::QwenKvHasElementBytes(st))
        Check(gufo::hip::QwenKvIsAllocatableByExecutor(st),
              "having element bytes implies being allocatable");
    }
    // And the converse is FALSE, which is why these are two predicates.
    Check(
        gufo::hip::QwenKvIsAllocatableByExecutor(QwenKvCacheStorage::kTurbo4) &&
            !gufo::hip::QwenKvHasElementBytes(QwenKvCacheStorage::kTurbo4),
        "allocatable does NOT imply element bytes — Turbo4 is the witness");
    // Holding a plane and being able to run against it are separate. Without
    // this split, flipping the first would leave an engine whose attention
    // reads a null fp16 member.
    Check(!gufo::hip::QwenKvIsExecutableByExecutor(QwenKvCacheStorage::kTurbo4),
          "a Turbo4 plane is held but NOT executable — no kernel takes a block "
          "pitch");
    Check(
        gufo::hip::QwenKvIsExecutableByExecutor(QwenKvCacheStorage::kFp16) &&
            gufo::hip::QwenKvIsExecutableByExecutor(QwenKvCacheStorage::kFp32),
        "fp16 and fp32 are executable as before");
  }

  // ---- 6. THE REAL TU NO LONGER CARRIES THE FALLTHROUGH ----------------
  // Probe #4 (enum 99 -> 4 B/element) was fixed in the HEADER, but arena.cpp
  // carried two more copies of the same ternary, and one of them was a live
  // correctness bug: the snapshot transfer took the fp32 branch for a Turbo4
  // plane and copied d_kv_f32_ with an fp32 pitch over 66-byte blocks.
  //
  // arena.cpp is HIP-only, so this asserts on its source and the offline hipcc
  // compile is the real oracle for the code itself.
  {
    std::FILE* f = std::fopen("src/models/qwen/hip/arena.cpp", "rb");
    std::string src;
    if (f != nullptr) {
      char buf[8192];
      std::size_t n = 0;
      while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0)
        src.append(buf, n);
      std::fclose(f);
    }
    Check(!src.empty(), "arena.cpp is readable for the port check");

    const std::string ternary = "kv_storage == QwenKvCacheStorage::kFp16 ?";
    const std::string ternary2 = "kv_storage_ == QwenKvCacheStorage::kFp16 ?";
    Check(src.find(ternary) == std::string::npos,
          "the layout site no longer has its own element-size ternary");
    // The one remaining kv_storage_ ternary is the SOURCE POINTER choice,
    // which is legitimately fp16-vs-fp32 now that block storage refuses above.
    std::size_t ptr_sites = 0, pos = 0;
    while ((pos = src.find(ternary2, pos)) != std::string::npos) {
      ++ptr_sites;
      pos += ternary2.size();
    }
    Check(ptr_sites <= 1,
          "only the fp16-vs-fp32 SOURCE POINTER choice remains, never an "
          "element-size fallback");
    Check(src.find("QwenKvElementBytes") != std::string::npos,
          "arena.cpp takes element bytes from the single authority");
    Check(src.find("QwenKvStorageIsKnown") != std::string::npos,
          "arena.cpp refuses an unknown storage before copying");
    Check(src.find("QwenKvIsBlockAddressed") != std::string::npos,
          "arena.cpp branches on block addressing for the plane size");
    // REVIEW: the block plane is now WIRED, so this is no longer a
    // refusal site. The old assertion here required the string "a Turbo4 plane
    // is REFUSED"; that requirement is retired because the thing it demanded is
    // now present. What replaced it is a stronger claim: the plane is allocated
    // from a member of its own, at a pitch the copy also uses.
    Check(src.find("hipMalloc(&d_kv_turbo4") != std::string::npos,
          "the Turbo4 plane is allocated into its OWN member, not aliased onto "
          "the fp16 or fp32 pointer");
    // Execution is refused from executor.hpp, not arena.cpp: the arena can HOLD
    // a block plane, and the entry points that would RUN against it are the
    // executor's. Asserting it in arena.cpp would be asserting the wrong file
    // and would pass vacuously.
    std::string exec_hdr;
    {
      std::FILE* xf = std::fopen("src/models/qwen/hip/executor.hpp", "rb");
      if (xf != nullptr) {
        char xb[32768];
        std::size_t xn = 0;
        while ((xn = std::fread(xb, 1, sizeof(xb), xf)) > 0)
          exec_hdr.append(xb, xn);
        std::fclose(xf);
      }
    }
    // The decode dev switch added a phase argument, so this literal changed from
    // RequireExecutableKvStorage(policy_.kv_cache_storage) to the two-argument
    // form at executor.hpp:635. A worker loosened the match to a bare
    // prefix to keep it green; review
    // REJECTED that, because a prefix match also accepts a malformed argument
    // such as policy_.kv_cache_storagey, so it weakens a guard to accommodate a
    // change. The property is unchanged, so assert the exact CURRENT call
    // instead. When the signature changes again, update this literal
    // deliberately and read the diff: that is the point of an exact guard, not
    // a defect of one.
    Check(
        exec_hdr.find(
            "RequireExecutableKvStorage(policy_.kv_cache_storage, phase)") !=
            std::string::npos,
        "the executor gates every public entry on an executable KV storage, so "
        "a held block plane is never run against by a kernel that cannot read "
        "it");
    Check(exec_hdr.find("void* d_kv_turbo4{nullptr};") != std::string::npos,
          "the arena member exists, which is what makes allocation legitimate");
    // The ternaries that chose the block plane by `== kTurbo4` now ask
    // QwenKvIsBlockAddressed(...), so Turbo5/Turbo6 take the same branch. A
    // site counts if it names kTurbo4 (the switch cases) or asks the predicate.
    std::size_t switches = 0, spos = 0;
    while ((spos = src.find("QwenKvCacheStorage::kTurbo4", spos)) !=
           std::string::npos) {
      ++switches;
      spos += 24;
    }
    spos = 0;
    while ((spos = src.find("QwenKvIsBlockAddressed(", spos)) !=
           std::string::npos) {
      ++switches;
      spos += 23;
    }
    Check(switches >= 6,
          "kTurbo4 is named at every site that must choose a layout: "
          "allocation, capture, restore, compact transfer, compact restore and "
          "the snapshot free");

    // ONE authority for the whole capture path. The path was wrong in FOUR
    // independent places (tail sizing, allocation, copy pitch, layout), and
    // fixing one of them would have left three answering differently.
    std::size_t guards = 0, pos2 = 0;
    while ((pos2 = src.find("RequireSnapshotCapableStorage(", pos2)) !=
           std::string::npos) {
      ++guards;
      pos2 += 26;
    }
    // 1 definition + 5 call sites: sizing, capture, the compact transfer, and
    // BOTH restore paths. The restores were unguarded — a save that refuses
    // followed by a restore that does not is worse than neither.
    Check(guards == 6,
          "all FIVE snapshot entry points call ONE shared storage authority");
    Check(
        src.find("void QwenGpuArena::RestoreSnapshot(") != std::string::npos &&
            src.find("void QwenGpuArena::RestoreCompactSnapshot(") !=
                std::string::npos,
        "both restore paths exist and are the ones being guarded");
    Check(src.find("SnapshotPayloadBytes(") != std::string::npos &&
              src.find("SaveSnapshot(") != std::string::npos,
          "the sizing and the capture entry points are both guarded");
    // The old inline duplicate must be gone.
    Check(src.find("Qwen compact snapshot has no block-addressed KV source") ==
              std::string::npos,
          "the inline duplicate guard is gone, replaced by the shared one");
    // And the sizing expression must no longer branch only on fp16. It now
    // measures a whole TOKEN through the same authority the copy uses — the
    // tail is a token count, and a Turbo4 token is 528 B, not 2 per value.
    Check(src.find("QwenKvBytesPerToken(policy_.kv_cache_storage, geometry)") !=
              std::string::npos,
          "the snapshot tail is sized from the per-token authority, which can "
          "express a block plane");
    Check(src.find("2 * sizeof(std::uint16_t)") == std::string::npos,
          "the old bytes-per-ELEMENT tail expression is gone from the sizing "
          "path");
    // All four copy sites describe themselves from ONE function. Three
    // independent derivations of a layout is three chances to disagree, and two
    // agreeing is worse than none.
    std::size_t describe = 0, dpos = 0;
    while ((dpos = src.find("QwenKvDescribeCopy(", dpos)) !=
           std::string::npos) {
      ++describe;
      dpos += 18;
    }
    Check(
        describe == 4,
        "all FOUR copy sites — capture, restore, compact transfer and compact "
        "restore — describe the copy with the one shared function");

    // 30b: the ALLOCATION was the root. `UsesFp16AttentionKv()` is
    // `storage == kFp16`, so selecting kTurbo4 allocated an FP32-SIZED buffer
    // while the rest of the engine believed it held Turbo4 blocks — 7.757576x
    // too many bytes per value, in the memory-SAFE direction.
    Check(src.find("Qwen executor cannot allocate a Turbo4 KV plane: a head is "
                   "not a ") != std::string::npos,
          "the KV ALLOCATION refuses a geometry whose head straddles a block");
    Check(src.find("Qwen executor KV storage is not a known precision") !=
              std::string::npos,
          "the KV allocation refuses an unknown precision");
    // Match the CALL, not the prose: the old expression still appears inside
    // the comment that explains why it was wrong.
    Check(src.find("hipMalloc(&d_kv_cache, total_kv") == std::string::npos,
          "the old unconditional fp32 KV allocation is gone");
    Check(src.find("hipMalloc(&d_attention_kv_f16, total_kv") ==
              std::string::npos,
          "the old fp16 KV allocation is gone too, both now sized by the "
          "single authority");
    Check(src.find("QwenKvBytesForElements(kv_storage, total_kv)") !=
              std::string::npos,
          "the allocation takes its byte count from the single authority");

    // The ESTIMATOR was the third instance, and the one that made the three
    // size computations disagree for Turbo4. Now converted too.
    Check(src.find("Qwen memory estimator cannot size a KV plane") !=
              std::string::npos,
          "the MEMORY ESTIMATOR refuses rather than omitting a KV plane it "
          "cannot size");
    Check(src.find("QwenKvBytesForElements(state_kv_storage, total_kv)") !=
              std::string::npos,
          "the estimator sizes the KV state from the single authority");
    Check(src.find("AddAllocation(CheckedMultiply(total_kv, 2),") ==
              std::string::npos,
          "the estimator no longer multiplies elements by a per-element size, "
          "which cannot express a block plane");

    // ZERO fp16 ternaries remain in the copy path. The previous count of three
    // was acceptable only because the guard ran first; with the block plane
    // wired there is no guard to hide behind, so the ternaries had to go.
    std::size_t ternary3 = 0, tpos = 0;
    while ((tpos = src.find("UsesFp16AttentionKv() ? sizeof", tpos)) !=
           std::string::npos) {
      ++ternary3;
      tpos += 20;
    }
    std::size_t guards2 = 0, gpos = 0;
    while ((gpos = src.find("RequireSnapshotCapableStorage(", gpos)) !=
           std::string::npos) {
      ++guards2;
      gpos += 26;
    }
    Check(
        ternary3 == 0,
        "no fp16 ternary survives in the copy path — every layout question is "
        "asked of one function now");
    Check(
        guards2 == 6,
        "and all five entry points plus the definition still call the storage "
        "authority, so an unknown precision cannot reach a copy");

    // And the predicate that caused it must not be load-bearing on its own.
    std::string policy;
    {
      std::FILE* pf =
          std::fopen("src/models/qwen/hip/execution_policy.hpp", "rb");
      if (pf != nullptr) {
        char pb[16384];
        std::size_t pn = 0;
        while ((pn = std::fread(pb, 1, sizeof(pb), pf)) > 0)
          policy.append(pb, pn);
        std::fclose(pf);
      }
    }
    Check(policy.find("UsesFp16AttentionKv") != std::string::npos,
          "the fp16 predicate still exists for callers that mean fp16");
    Check(policy.find("kTurbo4") != std::string::npos,
          "the policy still documents kTurbo4 as a storage");
  }

  std::printf(
      g_failures == 0 ? "\nALL CHECKS PASSED\n" : "\n%d CHECK(S) FAILED\n",
      g_failures);
  return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}