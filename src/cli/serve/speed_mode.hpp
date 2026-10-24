#pragma once

// `gufo serve --speed-mode exact|fast` and `--speed-levers a,b`: opt-in
// measured speedups that are NOT bit-exact with the default path. exact (the
// default) changes nothing. fast turns on every lever marked in_fast;
// --speed-levers turns on exactly the named levers; both = the union. Each
// active lever prints one line with its measured gain and output change, and
// is listed, with those numbers, in GET /health and GET /v1/models.
//
// A lever env the user set before launch always wins over the mode.
//
// Adding a lever = one row in kSpeedLevers. Numbers are copied from the run
// named in `source`; re-measure before changing them.

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gufo::server {

struct SpeedLever {
  std::string_view name;
  std::string_view env;       // GUFO_DEV_* switch; empty = flag lever
  std::string_view env_value;  // the value the lever sets ("1" for on/off)
  std::size_t draft_tokens;   // flag lever: --draft-tokens value (DFlash2 only)
  bool in_fast;
  std::string_view what;
  std::string_view gain;
  std::string_view kld;       // KLD vs exact + same-top1
  std::string_view source;
};

inline constexpr std::array<SpeedLever, 4> kSpeedLevers{{
    // gdn_chunk and draft6 are NOT in fast: each costs json decode ~17% on
    // the apex model; fast = the one lever with no code/json loss.
    {"gdn_chunk", "GUFO_DEV_GDN_CHUNK", "1", 0, false,
     "chunked WMMA DeltaNet prefill scan",
     "prefill +2.42%; decode prose +3.1%, code -1.0%, json -17.7%",
     "KLD vs exact 0.00103",
     "27B apex model, pin 1700"},
    // VRD6 is NOT in fast: verify -4.7% ms but decode -0.9% (27B apex model, pin 1700), so it
    // costs exactness for no decode gain. Listed so a user-set
    // GUFO_DEV_VRD6=1 is still reported as a non-exact lever.
    {"vrd6", "GUFO_DEV_VRD6", "1", 0, false,
     "Turbo6 verify attention reader",
     "verify -4.7% ms, decode -0.9%",
     "KLD vs exact 0.0006-0.0015 same-top1 97.3-99.6%",
     "27B apex model, pin 1700"},
    {"draft6", "", "", 6, false,
     "DFlash2 --draft-tokens 6 (default 7)",
     "decode prose +4.1%, code -10.1%, json -17.2%",
     "KLD vs exact 0.0005-0.0013 same-top1 96.9-99.1%; "
     "with gdn_chunk 0.00110 same-top1 98.3%",
     "27B apex model, pin 1700"},
    // Lookup route: rounds whose draft comes from repeated prompt text verify
    // at another width, so (like draft6) the logits change. Gain depends on
    // how much text repeats: code/json text stayed identical to exact.
    {"lookup_route", "GUFO_DEV_LOOKUP_ROUTE", "16", 0, true,
     "draft from repeated text (lookup) for up to 16 tokens",
     "decode prose +15.7%, code -0.2%, json 0.0% "
     "(code/json text identical to exact)",
     "KLD not measured; temp-0 prose text changes",
     "27B apex model, pin 1700"},
}};

// The bit-exact DFlash2 width: serve's --draft-tokens default.
inline constexpr std::size_t kExactDraftTokens = 7;

struct SpeedModePlan {
  std::vector<std::pair<std::string, std::string>> set_env;  // to setenv
  std::optional<std::size_t> draft_tokens;  // override --draft-tokens
  std::vector<std::string> active;          // non-exact levers in effect
  std::vector<std::string> measured;        // per active: gain | quality
  std::vector<std::string> lines;           // startup log lines
  std::string mode;                         // exact, fast or levers
  std::string error;                        // set: refuse to start
};

struct SpeedModeRequest {
  bool fast = false;           // --speed-mode fast
  std::string_view levers;     // --speed-levers, comma list of lever names
};

// Fills `error` (and nothing else) on an unknown lever name.
// getenv: the process env (a fake in tests). user_draft: the --draft-tokens
// value when the user passed it. dflash: DFlash2 is the speculative backend
// (draft6 was measured only there).
inline SpeedModePlan ResolveSpeedMode(const SpeedModeRequest& request,
                                      const char* (*getenv)(const char*),
                                      std::optional<std::size_t> user_draft,
                                      bool dflash) {
  SpeedModePlan plan;
  std::vector<std::string_view> named;
  for (std::string_view rest = request.levers; !rest.empty();) {
    const auto comma = rest.find(',');
    const auto name = rest.substr(0, comma);
    rest = comma == std::string_view::npos ? "" : rest.substr(comma + 1);
    bool known = false;
    for (const auto& lever : kSpeedLevers)
      known = known || lever.name == name;
    if (!known) {
      plan.error = "unknown --speed-levers name '" + std::string(name) +
                   "'; valid:";
      for (const auto& lever : kSpeedLevers)
        plan.error += " " + std::string(lever.name);
      return plan;
    }
    named.push_back(name);
  }
  const auto wanted = [&](const SpeedLever& lever) {
    return (request.fast && lever.in_fast) ||
           std::find(named.begin(), named.end(), lever.name) != named.end();
  };
  plan.mode = request.fast ? "fast" : named.empty() ? "exact" : "levers";
  const std::string mode = "speed-mode " + plan.mode + ": ";
  if (!named.empty()) {
    std::string list;
    for (const auto& lever : kSpeedLevers)
      if (wanted(lever))
        list += (list.empty() ? "" : ",") + std::string(lever.name);
    plan.lines.push_back(mode + "requested " + list +
                         (request.fast ? " (fast bundle + --speed-levers)"
                                       : " (--speed-levers)"));
  }
  const auto measured = [](const SpeedLever& lever) {
    return "measured " + std::string(lever.gain) + " | " +
           std::string(lever.kld) + " (" + std::string(lever.source) + ")";
  };
  const auto describe = [&](const SpeedLever& lever) {
    return std::string(lever.name) + " on: " + std::string(lever.what) +
           " | " + measured(lever);
  };
  const auto activate = [&](const SpeedLever& lever) {
    plan.active.emplace_back(lever.name);
    plan.measured.push_back(measured(lever));
  };
  for (const auto& lever : kSpeedLevers) {
    const bool want = wanted(lever);
    if (lever.env.empty()) {  // the draft-width lever
      if (!dflash) {
        if (want)
          plan.lines.push_back(mode + std::string(lever.name) +
                               " off: needs --speculative dflash2");
        continue;
      }
      if (user_draft && *user_draft != kExactDraftTokens) {
        const std::string n = std::to_string(*user_draft);
        plan.active.push_back("draft" + n);
        plan.measured.push_back(*user_draft == lever.draft_tokens
                                    ? measured(lever)
                                    : "not measured");
        plan.lines.push_back(mode + "draft width " + n +
                             " set by user: not bit-exact with the default " +
                             std::to_string(kExactDraftTokens));
      } else if (want && user_draft) {
        plan.lines.push_back(mode + std::string(lever.name) +
                             " off: --draft-tokens given by user wins");
      } else if (want) {
        plan.draft_tokens = lever.draft_tokens;
        activate(lever);
        plan.lines.push_back(mode + describe(lever));
      }
      continue;
    }
    const std::string env(lever.env);
    if (const char* user = getenv(env.c_str()); user != nullptr) {
      const std::string user_env = "user env " + env + "=" + user;
      if (std::string_view(user) != "0" && *user != '\0') {
        activate(lever);
        plan.lines.push_back(mode + describe(lever) + " [set by " + user_env +
                             "]");
      } else if (want) {
        plan.lines.push_back(mode + std::string(lever.name) + " off: " +
                             user_env + " wins");
      }
      continue;
    }
    if (want) {
      plan.set_env.emplace_back(env, std::string(lever.env_value));
      activate(lever);
      plan.lines.push_back(mode + describe(lever));
    }
  }
  return plan;
}

}  // namespace gufo::server
