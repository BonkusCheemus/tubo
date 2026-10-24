// CPU test of --speed-mode / --speed-levers resolution
// (src/cli/serve/speed_mode.hpp).
#include "src/cli/serve/speed_mode.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {
using gufo::server::ResolveSpeedMode;
using gufo::server::SpeedModePlan;
using gufo::server::SpeedModeRequest;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

const char* gdn_value = nullptr;  // fake env: only the lever switches
const char* vrd6_value = nullptr;
const char* route_value = nullptr;
const char* FakeEnv(const char* name) {
  if (std::strcmp(name, "GUFO_DEV_GDN_CHUNK") == 0)
    return gdn_value;
  if (std::strcmp(name, "GUFO_DEV_VRD6") == 0)
    return vrd6_value;
  if (std::strcmp(name, "GUFO_DEV_LOOKUP_ROUTE") == 0)
    return route_value;
  return nullptr;
}
bool Has(const SpeedModePlan& plan, std::string_view lever) {
  return std::find(plan.active.begin(), plan.active.end(), lever) !=
         plan.active.end();
}
bool Sets(const SpeedModePlan& plan, std::string_view env) {
  return std::any_of(plan.set_env.begin(), plan.set_env.end(),
                     [&](const auto& kv) { return kv.first == env; });
}
SpeedModePlan Run(SpeedModeRequest request,
                  std::optional<std::size_t> user_draft = std::nullopt,
                  bool dflash = true) {
  return ResolveSpeedMode(request, FakeEnv, user_draft, dflash);
}

void TestExactChangesNothing() {
  gdn_value = vrd6_value = route_value = nullptr;
  for (const auto user_draft : {std::optional<std::size_t>{},
                                std::optional<std::size_t>{7u}})
    for (const bool dflash : {true, false}) {
      const auto plan = Run({}, user_draft, dflash);
      Expect(plan.set_env.empty() && !plan.draft_tokens &&
                 plan.active.empty() && plan.measured.empty() &&
                 plan.lines.empty() && plan.mode == "exact" &&
                 plan.error.empty(),
             "exact with no env must set no env, change nothing, print "
             "nothing");
    }
}

void TestFastSetsBundle() {
  gdn_value = vrd6_value = route_value = nullptr;
  const auto plan = Run({.fast = true, .levers = ""});
  Expect(plan.set_env.size() == 1 &&
             plan.set_env[0] == std::pair<std::string, std::string>(
                                    "GUFO_DEV_LOOKUP_ROUTE", "16"),
         "fast sets only GUFO_DEV_LOOKUP_ROUTE=16");
  Expect(!plan.draft_tokens, "fast keeps the bit-exact draft width");
  Expect(plan.active == std::vector<std::string>{"lookup_route"} &&
             plan.measured.size() == 1,
         "fast active list = lookup_route only");
  Expect(plan.lines.size() == 1 &&
             plan.lines[0].starts_with("speed-mode fast: lookup_route on: ") &&
             plan.lines[0].find("decode prose +15.7%, code -0.2%, json 0.0%") !=
                 std::string::npos &&
             plan.lines[0].find("(27B apex model, pin 1700)") !=
                 std::string::npos &&
             plan.measured[0].starts_with("measured decode prose +15.7%"),
         "fast prints one line with the measured numbers");
}

void TestUserEnvWins() {
  gdn_value = vrd6_value = nullptr;
  route_value = "0";
  auto plan = Run({.fast = true, .levers = ""});
  Expect(plan.set_env.empty() && !Has(plan, "lookup_route"),
         "user GUFO_DEV_LOOKUP_ROUTE=0 beats fast");
  Expect(plan.lines.size() == 1 &&
             plan.lines[0].find("user env GUFO_DEV_LOOKUP_ROUTE=0 wins") !=
                 std::string::npos,
         "user override is printed");
  route_value = nullptr;
  gdn_value = "0";
  plan = Run({.levers = "gdn_chunk"});
  Expect(plan.set_env.empty() && !Has(plan, "gdn_chunk"),
         "user GUFO_DEV_GDN_CHUNK=0 beats --speed-levers");
  gdn_value = nullptr;
  vrd6_value = "1";
  plan = Run({});
  Expect(plan.set_env.empty() && Has(plan, "vrd6") && plan.lines.size() == 1,
         "exact still reports a user-set non-exact lever");
  vrd6_value = nullptr;
  route_value = "8";
  plan = Run({});
  Expect(plan.set_env.empty() && Has(plan, "lookup_route") &&
             plan.lines.size() == 1,
         "exact reports a user LOOKUP_ROUTE=8 as non-exact");
  route_value = nullptr;
}

void TestDraftRule() {
  gdn_value = vrd6_value = nullptr;
  auto plan = Run({.fast = true, .levers = ""});
  Expect(!plan.draft_tokens && !Has(plan, "draft6") && !Has(plan, "gdn_chunk"),
         "fast leaves draft6 and gdn_chunk off");
  plan = Run({.levers = "draft6"}, 7u);
  Expect(!plan.draft_tokens && !Has(plan, "draft6"),
         "user --draft-tokens 7 beats --speed-levers draft6");
  plan = Run({}, 7u);
  Expect(plan.active.empty() && plan.lines.empty(),
         "exact with user -d 7 is the exact path");
  plan = Run({}, 6u);
  Expect(!plan.draft_tokens && Has(plan, "draft6") && plan.lines.size() == 1 &&
             plan.lines[0] == "speed-mode exact: draft width 6 set by user: "
                              "not bit-exact with the default 7",
         "user -d 6 under exact is reported as non-exact");
  plan = Run({.fast = true, .levers = ""}, 5u);
  Expect(!plan.draft_tokens && Has(plan, "draft5") && !Has(plan, "draft6") &&
             plan.measured.size() == plan.active.size() &&
             plan.measured[0] == "not measured",
         "user -d 5 under fast: reported as not measured");
  plan = Run({.levers = "draft6"}, std::nullopt, false);
  Expect(!plan.draft_tokens && !Has(plan, "draft6"),
         "draft6 only under DFlash2");
}

void TestLevers() {
  gdn_value = vrd6_value = nullptr;
  auto plan = Run({.levers = "vrd6"});
  Expect(plan.mode == "levers" && plan.set_env.size() == 1 &&
             Sets(plan, "GUFO_DEV_VRD6") && plan.active.size() == 1 &&
             Has(plan, "vrd6") && !plan.draft_tokens,
         "single lever turns on exactly that lever");
  plan = Run({.levers = "gdn_chunk,vrd6"});
  Expect(plan.set_env.size() == 2 && Has(plan, "gdn_chunk") &&
             Has(plan, "vrd6") && !Has(plan, "draft6"),
         "list turns on exactly the listed levers");
  plan = Run({.levers = "draft6"});
  Expect(plan.set_env.empty() && plan.draft_tokens == 6u &&
             plan.active.size() == 1,
         "draft6 alone");
  plan = Run({.fast = true, .levers = "vrd6"});
  Expect(plan.mode == "fast" && plan.active.size() == 2 &&
             Has(plan, "lookup_route") && Has(plan, "vrd6") &&
             plan.lines[0] == "speed-mode fast: requested vrd6,lookup_route "
                              "(fast bundle + --speed-levers)",
         "fast + --speed-levers = union, printed");
  plan = Run({.fast = true, .levers = "gdn_chunk,draft6"});
  Expect(plan.set_env.size() == 2 && Sets(plan, "GUFO_DEV_GDN_CHUNK") &&
             Sets(plan, "GUFO_DEV_LOOKUP_ROUTE") && plan.draft_tokens == 6u &&
             plan.active.size() == 3 && plan.measured.size() == 3,
         "the old bundle stays reachable by name");
  plan = Run({.levers = "gdn_chunk,turbo9"});
  Expect(plan.error == "unknown --speed-levers name 'turbo9'; valid: "
                       "gdn_chunk vrd6 draft6 lookup_route" &&
             plan.set_env.empty() && plan.active.empty(),
         "unknown name refuses with the valid list");
  Expect(!Run({.levers = "gdn_chunk,,vrd6"}).error.empty(),
         "empty name refuses");
}
}  // namespace

int main() {
  TestExactChangesNothing();
  TestFastSetsBundle();
  TestUserEnvWins();
  TestDraftRule();
  TestLevers();
  std::cout << "speed_mode_test: ok\n";
  return 0;
}
