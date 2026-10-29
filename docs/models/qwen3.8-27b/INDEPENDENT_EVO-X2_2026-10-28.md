# Independent report: Strix Halo 64 GB, gufo vs tubo, Qwen3.8-27B (EVO-X2, 2026-10-28)

Independent report by u/bob310 (Reddit, r/StrixHalo), included with permission. Report generated with Claude by its author. Machine: GMKtec EVO-X2, 64 GB. Tested tubo @ 2a258b9. Translated from French; the original follows the translation and wins on any difference.

## Translation (English)

### Strix Halo 64 GB — gufo versus tubo, Qwen3.8-27B (evo2, 2026-10-28)

#### System

| Item | Value |
| --- | --- |
| Machine | GMKtec EVO-X2 (BIOS "EVO-X2S 1.13") |
| APU | AMD Ryzen AI MAX+ 395, Radeon 8060S (gfx1151), 16C/32T |
| Memory | 64 GB LPDDR5X-8000 (8 × 8 GB), UMA buffer 2 GB in the BIOS |
| GPU-addressable memory | 54 GiB GTT (amdgpu.gttsize=55296 ttm.pages_limit=14155776 ttm.page_pool_size=14155776) |
| Storage | EXT X200E NVMe, PCIe 4.0 x4 — 7.0 GB/s sequential read, 73k IOPS 4k random read (fio, QD64×4) |
| OS | Ubuntu 26.04.1 LTS (server) |
| Kernel | 7.3.0-next-20261028-perfopt (linux-next 2026-10-28, AMD IOMMU PerfOpt armed); the baseline works on 7.0.0-42-generic |
| Vulkan | Mesa 26.0.8 RADV |
| Power | original BIOS performance mode, CPU governor powersave / EPP balance_performance, GPU DPM auto (no frequency pinning) |

#### Engines and model

| Item | Value |
| --- | --- |
| Model | Qwen3.8-27B, Unsloth UD-Q4_K_XL (17.6 GB) + z-lab DFlash2 drafter Q4_K_M (gufo/tubo); Unsloth UD-Q6_K for llama.cpp |
| gufo 0.13.0 | engine fcf8bf7, toolkits 0ff269d (container built with Nix, ROCm version not recorded) |
| gufo 0.26.0 | engine 78d7418, toolkits 6137af8, ROCm 10.1 |
| gufo base of tubo | engine 1b8ad798, toolkits 19354d07, ROCm 7.2 |
| tubo | BonkusCheemus/tubo @ 2a258b9 + a local CMake fix (evaluation targets guarded with if(EXISTS), they are filtered out of the Nix source), toolkits 19354d07, ROCm 7.2 |
| llama.cpp | Vulkan fork by Nathanw b02cb35 |
| Server parameters | --sessions 1, --think off (quality runs), --cache-ram-bytes 536870912, hardened Docker container (read-only, all capabilities dropped) |
| "draft flags" | --draft-tokens 7 --min-draft-tokens 1 --draft-policy fixed |
| tubo environment | docs/tubo.env minus NPU, EMBED_SSD and IQ3-only lines (the model is Q4_K_XL); "fp16 KV" = also minus each TURBO line |

#### Results following REPRODUCE.md (sections 4 and 5)

GPU frequency pinned at 2000 MHz (pp_od_clk_voltage s0 = s1 = 2000, free afterwards). Vanilla = gufo 1b8ad798, tubo = 2a258b9 (+ the CMake guard above), all built with ROCm 7.2 through the gufo Nix toolchain (toolkits 19354d07) and run in Docker. No NPU stack: GUFO_NPU_\* and GUFO_DEV_EMBED_SSD lines removed from docs/tubo.env.

Section 4 — A/B server with live_ab.py (ab_prompt.txt with a unique first line, --context 4096 --think off, --max-tokens 512, temperature 0; 1 warm-up discarded + 5 measured requests, server restarted, repeated once → 10 measured requests per engine; tubo with the full environment of section 3, that is turbo6 KV, plus the draft flags)

| Engine | Decode tok/s (mean ± standard deviation) | Prefill tok/s | Time per request |
| --- | --- | --- | --- |
| vanilla gufo 1b8ad798 | 34.50 ± 1.25 | 327.6 ± 0.7 | 15.97 ± 0.69 s |
| tubo | 40.40 ± 2.14 (+17.1 %) | 333.0 ± 0.3 (+1.6 %) | 13.53 ± 0.55 s (−15.3 %) |

Section 5 — engine-only A/B with upstream  model-bench.py  (bench.json from vanilla, --cache-ram-bytes 2147483648, table single-dflash2-q4, depth 0, context 4096, 1 warm-up discarded + 3 repetitions per arm; tubo with tubo-fp16kv.env and the draft flags)

| Cell | vanilla pp / tg | tubo pp / tg | delta tg |
| --- | --- | --- | --- |
| single-dflash2-q4 | 485.4 / 23.33 | 487.3 / 24.24 | +3.9 % |
| single-dflash2-repetition-q4 | 483.4 / 65.98 (6.11 accepted/step) | 486.9 / 63.43 (5.64 accepted/step) | −3.9 % |

Reading: with fp16 KV, the engine gains are within a few percent on either side; the section 4 gain (+17 %) comes from turbo6 KV plus the draft flags. This 64 GB mini-PC shows about two thirds of the +28 % reported on the 32 GB laptop.

#### Decode speed, short prompts — our own method (not REPRODUCE.md; single stream, 12 interleaved runs, 32k context, frequencies not pinned)

Request: "write a Python function + 3 unittest tests", ~400–650 output tokens, thinking off.

| Engine / configuration | Mean tok/s | Min–max | vs base |
| --- | --- | --- | --- |
| gufo 0.13.0 | 58.1 | 50.6–67.1 | −0.3 % |
| gufo base 1b8ad798 | 58.3 | 52.1–62.3 | — |
| gufo 0.13.0 + draft flags | 60.3 | 57.0–64.9 | +3.4 % |
| gufo base + draft flags | 61.1 | 55.7–65.6 | +4.8 % |
| tubo, fp16 KV + draft flags | 63.4 | 56.4–69.8 | +8.7 % |
| tubo, turbo6 KV + draft flags | 64.7 | 59.8–72.4 | +11.0 % |
| gufo 0.26.0 (5 runs, separate session) | 60.2 | 58.7–62.8 | — |

#### Quality and long context (gufo 0.13.0 + draft flags versus tubo turbo6 + draft flags)

| Test | gufo 0.13.0 | tubo turbo6 |
| --- | --- | --- |
| HumanEval pass@1 (164, temp 0, tubo's  humaneval.py , scored in a container with no network) | 156/164 (95.1 %) | 155/164 (94.5 %) |
| HumanEval decode, overall / median | 52.3 / 67.9 tok/s | 56.4 / 73.4 tok/s (+8 %) |
| Needle, 32k context, seeds 1 and 2 (tubo's qual2.py) | 20/20, 20/20 | 20/20, 20/20 |
| Decode at ~30.6k prompt tokens | 33.9 tok/s | 46.1 tok/s (+36 %) |
| Prefill at ~30.6k prompt tokens | 567 tok/s | 573 tok/s |

#### Other measurements on the same machine

| Test | Result |
| --- | --- |
| llama.cpp Vulkan, Q6_K, kernel 7.0 → 7.3-perfopt | pp512 406 → 437 (+7.6 %), pp2048 396 → 429 (+8.3 %), tg128 10.06 → 10.16 |
| gufo 0.13 DFlash2, kernel 7.0 → 7.3-perfopt | 59.7 → 59.0 tok/s (no change) |
| gufo --sessions 4 -c 65536, 1 / 2 / 4 concurrent streams | 32 / 50 / 54 tok/s in total (versus 59 single session) — parallelism does not pay off with DFlash2 |
| --sessions 4 -c 131072 | refused: "the text execution request exceeds the state capacity" |

#### Remarks

Same machine, same model files, one server at a time, no GPU frequency pinning, ambient temperature. The spread between runs is ±5 tok/s, so differences below ~5 % are inconclusive at n = 12.

About half of tubo's short-prompt gain comes from the draft flags, which original gufo also accepts. tubo's own engine gain is ~4 % in a short context but grows with context length (turbo6 KV halves the KV read per token).

tubo as published does not build through the gufo Nix toolchain (the CMake references tools/qwen27b/\*.hip that the Nix source filter removes) and requires ROCm 7.2.

## Original (French)

### Strix Halo 64 Go — gufo contre tubo, Qwen3.8-27B (evo2, 2026-10-28)

#### Système

| Élément | Valeur |
| --- | --- |
| Machine | GMKtec EVO-X2 (BIOS "EVO-X2S 1.13") |
| APU | AMD Ryzen AI MAX+ 395, Radeon 8060S (gfx1151), 16C/32T |
| Mémoire | 64 Go LPDDR5X-8000 (8 × 8 Go), buffer UMA 2 Go dans le BIOS |
| Mémoire adressable GPU | 54 GiB GTT (amdgpu.gttsize=55296 ttm.pages_limit=14155776 ttm.page_pool_size=14155776) |
| Stockage | EXT X200E NVMe, PCIe 4.0 x4 — 7.0 Go/s lecture séquentielle, 73k IOPS 4k lecture aléatoire (fio, QD64×4) |
| OS | Ubuntu 26.04.1 LTS (serveur) |
| Noyau | 7.3.0-next-20261028-perfopt (linux-next 2026-10-28, AMD IOMMU PerfOpt armé); la ligne de base fonctionne sur 7.0.0-42-generic |
| Vulkan | Mesa 26.0.8 RADV |
| Alimentation | mode performance BIOS d'origine, gouverneur CPU powersave / EPP balance_performance, DPM GPU auto (pas de fixation de fréquence) |

#### Moteurs et modèle

| Élément | Valeur |
| --- | --- |
| Modèle | Qwen3.8-27B, Unsloth UD-Q4_K_XL (17.6 Go) + z-lab DFlash2 redacteur Q4_K_M (gufo/tubo); Unsloth UD-Q6_K pour llama.cpp |
| gufo 0.13.0 | moteur fcf8bf7, boîtes à outils 0ff269d (conteneur construit avec Nix, version ROCm non enregistrée) |
| gufo 0.26.0 | moteur 78d7418, boîtes à outils 6137af8, ROCm 10.1 |
| gufo base de tubo | moteur 1b8ad798, boîtes à outils 19354d07, ROCm 7.2 |
| tubo | BonkusCheemus/tubo @ 2a258b9 + un correctif CMake local (cibles d'évaluation protégées avec if(EXISTS), elles sont filtrées à partir de la source Nix), boîtes à outils 19354d07, ROCm 7.2 |
| llama.cpp | fork Vulkan de Nathanw b02cb35 |
| Paramètres du serveur | --sessions 1, --think off (exécutions de qualité), --cache-ram-bytes 536870912, conteneur Docker durci (lecture seule, suppression de toutes les capacités) |
| "drapeaux de brouillon" | --draft-tokens 7 --min-draft-tokens 1 --draft-policy fixed |
| environnement tubo | docs/tubo.env moins NPU, EMBED_SSD et lignes uniquement IQ3 (le modèle est Q4_K_XL); "fp16 KV" = également moins chaque TURBO ligne |

#### Résultats suivant REPRODUCE.md (sections 4 et 5)

Fréquence GPU fixée à 2000 MHz (pp_od_clk_voltage s0 = s1 = 2000, libre ensuite). Vanilla = gufo 1b8ad798, tubo = 2a258b9 (+ le garde CMake ci-dessus), tous construits avec ROCm 7.2 à travers la chaîne d'outils gufo Nix (boîtes à outils 19354d07) et exécutés dans Docker. Pas de pile NPU : GUFO_NPU_\* et GUFO_DEV_EMBED_SSD lignes supprimées de docs/tubo.env.

Section 4 — serveur A/B avec live_ab.py (ab_prompt.txt avec une première ligne unique, --context 4096 --think off, --max-tokens 512, température 0 ; 1 échauffement jeté + 5 requêtes mesurées, serveur redémarré, répété une fois → 10 requêtes mesurées par moteur ; tubo avec l'environnement complet de la section 3, c'est-à-dire turbo6 KV, plus les drapeaux de brouillon)

| Moteur | Décryptage tok/s (moyenne ± écart-type) | Pré-remplissage tok/s | Temps par requête |
| --- | --- | --- | --- |
| vanilla gufo 1b8ad798 | 34.50 ± 1.25 | 327.6 ± 0.7 | 15.97 ± 0.69 s |
| tubo | 40.40 ± 2.14 (+17.1 %) | 333.0 ± 0.3 (+1.6 %) | 13.53 ± 0.55 s (−15.3 %) |

Section 5 — moteur uniquement A/B avec upstream  model-bench.py  (bench.json de vanilla, --cache-ram-bytes 2147483648, tableau single-dflash2-q4, profondeur 0, contexte 4096, 1 échauffement jeté + 3 répétitions par bras ; tubo avec tubo-fp16kv.env et les drapeaux de brouillon)

| Cellule | vanilla pp / tg | tubo pp / tg | delta tg |
| --- | --- | --- | --- |
| single-dflash2-q4 | 485.4 / 23.33 | 487.3 / 24.24 | +3.9 % |
| single-dflash2-repetition-q4 | 483.4 / 65.98 (6.11 acceptés/étape) | 486.9 / 63.43 (5.64 acceptés/étape) | −3.9 % |

Lecture : avec fp16 KV, les gains du moteur sont dans quelques pourcentages d'un côté ou de l'autre ; le gain section 4 (+17 %) provient de turbo6 KV plus les drapeaux de brouillon. Ce mini-PC de 64 Go montre environ deux tiers des +28 % rapportés sur le portable de 32 Go.

#### Vitesse de décryptage, prompts courts — notre propre méthode (pas REPRODUCE.md; flux unique, 12 exécutions entrecroisées, contexte 32k, fréquences non fixées)

Demande : "écrire une fonction Python + 3 tests unittest", ~400–650 jetons de sortie, réflexion éteinte.

| Moteur / configuration | Moyenne tok/s | Min–max | vs base |
| --- | --- | --- | --- |
| gufo 0.13.0 | 58.1 | 50.6–67.1 | −0.3 % |
| base gufo 1b8ad798 | 58.3 | 52.1–62.3 | — |
| gufo 0.13.0 + drapeaux de brouillon | 60.3 | 57.0–64.9 | +3.4 % |
| base gufo + drapeaux de brouillon | 61.1 | 55.7–65.6 | +4.8 % |
| tubo, fp16 KV + drapeaux de brouillon | 63.4 | 56.4–69.8 | +8.7 % |
| tubo, turbo6 KV + drapeaux de brouillon | 64.7 | 59.8–72.4 | +11.0 % |
| gufo 0.26.0 (5 exécutions, session séparée) | 60.2 | 58.7–62.8 | — |

#### Qualité et long contexte (gufo 0.13.0 + drapeaux de brouillon contre tubo turbo6 + drapeaux de brouillon)

| Test | gufo 0.13.0 | tubo turbo6 |
| --- | --- | --- |
| HumanEval pass@1 (164, temp 0, le  humaneval.py  de tubo, noté dans un conteneur sans réseau) | 156/164 (95.1 %) | 155/164 (94.5 %) |
| Décryptage HumanEval, global / médian | 52.3 / 67.9 tok/s | 56.4 / 73.4 tok/s (+8 %) |
| Needle, 32k contexte, graines 1 et 2 (le qual2.py de tubo) | 20/20, 20/20 | 20/20, 20/20 |
| Décryptage à ~30.6k jetons de prompt | 33.9 tok/s | 46.1 tok/s (+36 %) |
| Pré-remplissage à ~30.6k jetons de prompt | 567 tok/s | 573 tok/s |

#### Autres mesures sur la même machine

| Test | Résultat |
| --- | --- |
| llama.cpp Vulkan, Q6_K, noyau 7.0 → 7.3-perfopt | pp512 406 → 437 (+7.6 %), pp2048 396 → 429 (+8.3 %), tg128 10.06 → 10.16 |
| gufo 0.13 DFlash2, noyau 7.0 → 7.3-perfopt | 59.7 → 59.0 tok/s (pas de changement) |
| gufo --sessions 4 -c 65536, 1 / 2 / 4 flux concurrents | 32 / 50 / 54 tok/s au total (contre 59 session unique) — le parallélisme ne paye pas avec DFlash2 |
| --sessions 4 -c 131072 | refusé : "la demande d'exécution du texte dépasse la capacité de l'état" |

#### Remarques

Même machine, mêmes fichiers modèles, un serveur à la fois, pas de fixation de fréquence GPU, température ambiante. L'écart entre les exécutions est ±5 tok/s, donc les différences sous ~5 % ne sont pas concluantes à n = 12.

Environ la moitié du gain de prompt court du tubo provient des drapeaux de brouillon, que gufo d'origine accepte également. Le gain propre au moteur de tubo est ~4 % dans un contexte court mais augmente avec la longueur du contexte (turbo6 KV divise par deux la lecture KV par jeton).

Le tubo tel que publié ne se construit pas à travers la chaîne d'outils gufo Nix (les références CMake tools/qwen27b/\*.hip que le filtre de source Nix supprime) et nécessite ROCm 7.2.
