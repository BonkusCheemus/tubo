# 独立レポート: Strix Halo 64 GB、gufo と tubo の比較、Qwen3.8-27B(EVO-X2、2026-10-28)

u/bob310(Reddit、r/StrixHalo)による独立レポート。許可を得て掲載。レポートは作者が Claude で生成。マシン: GMKtec EVO-X2、64 GB。テスト対象 tubo @ 2a258b9。フランス語から翻訳。原文(フランス語)は英語版ファイルにあり、相違がある場合は原文が優先される。

## 翻訳(日本語)

### Strix Halo 64 GB — gufo 対 tubo、Qwen3.8-27B(evo2、2026-10-28)

#### システム

| 項目 | 値 |
| --- | --- |
| マシン | GMKtec EVO-X2(BIOS "EVO-X2S 1.13") |
| APU | AMD Ryzen AI MAX+ 395、Radeon 8060S(gfx1151)、16C/32T |
| メモリ | 64 GB LPDDR5X-8000(8 × 8 GB)、BIOS で UMA バッファ 2 GB |
| GPU がアドレス指定可能なメモリ | 54 GiB GTT(amdgpu.gttsize=55296 ttm.pages_limit=14155776 ttm.page_pool_size=14155776) |
| ストレージ | EXT X200E NVMe、PCIe 4.0 x4 — シーケンシャルリード 7.0 GB/s、73k IOPS 4k ランダムリード(fio、QD64×4) |
| OS | Ubuntu 26.04.1 LTS(サーバー) |
| カーネル | 7.3.0-next-20261028-perfopt(linux-next 2026-10-28、AMD IOMMU PerfOpt 有効化済み)。ベースラインは 7.0.0-42-generic で動作する |
| Vulkan | Mesa 26.0.8 RADV |
| 電力 | 元の BIOS パフォーマンスモード、CPU ガバナー powersave / EPP balance_performance、GPU DPM auto(周波数固定なし) |

#### エンジンとモデル

| 項目 | 値 |
| --- | --- |
| モデル | Qwen3.8-27B、Unsloth UD-Q4_K_XL(17.6 GB)+ z-lab DFlash2 drafter Q4_K_M(gufo/tubo)。llama.cpp には Unsloth UD-Q6_K |
| gufo 0.13.0 | エンジン fcf8bf7、ツールキット 0ff269d(Nix でビルドしたコンテナ、ROCm バージョンは未記録) |
| gufo 0.26.0 | エンジン 78d7418、ツールキット 6137af8、ROCm 10.1 |
| tubo のベースとなる gufo | エンジン 1b8ad798、ツールキット 19354d07、ROCm 7.2 |
| tubo | BonkusCheemus/tubo @ 2a258b9 + ローカルの CMake 修正(評価ターゲットを if(EXISTS) で保護。Nix ソースからフィルタされるため)、ツールキット 19354d07、ROCm 7.2 |
| llama.cpp | Nathanw による Vulkan フォーク b02cb35 |
| サーバーパラメータ | --sessions 1、--think off(品質テストの実行)、--cache-ram-bytes 536870912、堅牢化した Docker コンテナ(読み取り専用、すべての capability を削除) |
| "draft flags" | --draft-tokens 7 --min-draft-tokens 1 --draft-policy fixed |
| tubo の環境 | docs/tubo.env から NPU、EMBED_SSD、IQ3 専用の行を除いたもの(モデルは Q4_K_XL)。"fp16 KV" = さらに各 TURBO 行も除く |

#### REPRODUCE.md に従った結果(セクション 4 と 5)

GPU 周波数を 2000 MHz に固定(pp_od_clk_voltage s0 = s1 = 2000、その後は解放)。Vanilla = gufo 1b8ad798、tubo = 2a258b9(+ 上記の CMake ガード)。すべて ROCm 7.2 で gufo の Nix ツールチェーン(ツールキット 19354d07)を通してビルドし、Docker 内で実行。NPU スタックなし: GUFO_NPU_\* と GUFO_DEV_EMBED_SSD の行は docs/tubo.env から削除。

セクション 4 — live_ab.py を使ったサーバー A/B(先頭行が一意の ab_prompt.txt、--context 4096 --think off、--max-tokens 512、temperature 0。ウォームアップ 1 回は破棄 + 計測リクエスト 5 件、サーバー再起動、もう一度繰り返し → エンジンごとに計測リクエスト 10 件。tubo はセクション 3 の完全な環境、つまり turbo6 KV に加えて draft flags)

| エンジン | Decode tok/s(平均 ± 標準偏差) | Prefill tok/s | リクエストあたりの時間 |
| --- | --- | --- | --- |
| vanilla gufo 1b8ad798 | 34.50 ± 1.25 | 327.6 ± 0.7 | 15.97 ± 0.69 s |
| tubo | 40.40 ± 2.14 (+17.1 %) | 333.0 ± 0.3 (+1.6 %) | 13.53 ± 0.55 s (−15.3 %) |

セクション 5 — upstream の  model-bench.py  を使ったエンジンのみの A/B(vanilla の bench.json、--cache-ram-bytes 2147483648、テーブル single-dflash2-q4、深さ 0、コンテキスト 4096、アームごとにウォームアップ 1 回は破棄 + 3 回反復。tubo は tubo-fp16kv.env と draft flags)

| セル | vanilla pp / tg | tubo pp / tg | tg の差 |
| --- | --- | --- | --- |
| single-dflash2-q4 | 485.4 / 23.33 | 487.3 / 24.24 | +3.9 % |
| single-dflash2-repetition-q4 | 483.4 / 65.98(6.11 accepted/step) | 486.9 / 63.43(5.64 accepted/step) | −3.9 % |

所見: fp16 KV では、エンジンの向上は両方向とも数パーセント以内に収まる。セクション 4 の向上(+17 %)は turbo6 KV と draft flags によるもの。この 64 GB のミニ PC は、32 GB のノート PC で報告された +28 % の約三分の二を示す。

#### Decode 速度、短いプロンプト — 独自の方法(REPRODUCE.md ではない。単一ストリーム、12 回のインターリーブ実行、32k コンテキスト、周波数は固定せず)

リクエスト: "Python の関数 + 3 つの unittest テストを書いて"、出力は約 400–650 トークン、thinking はオフ。

| エンジン / 構成 | 平均 tok/s | 最小–最大 | vs base |
| --- | --- | --- | --- |
| gufo 0.13.0 | 58.1 | 50.6–67.1 | −0.3 % |
| gufo base 1b8ad798 | 58.3 | 52.1–62.3 | — |
| gufo 0.13.0 + draft flags | 60.3 | 57.0–64.9 | +3.4 % |
| gufo base + draft flags | 61.1 | 55.7–65.6 | +4.8 % |
| tubo、fp16 KV + draft flags | 63.4 | 56.4–69.8 | +8.7 % |
| tubo、turbo6 KV + draft flags | 64.7 | 59.8–72.4 | +11.0 % |
| gufo 0.26.0(5 回実行、別セッション) | 60.2 | 58.7–62.8 | — |

#### 品質と長いコンテキスト(gufo 0.13.0 + draft flags 対 tubo turbo6 + draft flags)

| テスト | gufo 0.13.0 | tubo turbo6 |
| --- | --- | --- |
| HumanEval pass@1(164、temp 0、tubo の  humaneval.py 、ネットワークなしのコンテナ内で採点) | 156/164 (95.1 %) | 155/164 (94.5 %) |
| HumanEval の decode、全体 / 中央値 | 52.3 / 67.9 tok/s | 56.4 / 73.4 tok/s (+8 %) |
| Needle、32k コンテキスト、シード 1 と 2(tubo の qual2.py) | 20/20, 20/20 | 20/20, 20/20 |
| プロンプト約 30.6k トークンでの decode | 33.9 tok/s | 46.1 tok/s (+36 %) |
| プロンプト約 30.6k トークンでの prefill | 567 tok/s | 573 tok/s |

#### 同じマシンでのその他の測定

| テスト | 結果 |
| --- | --- |
| llama.cpp Vulkan、Q6_K、カーネル 7.0 → 7.3-perfopt | pp512 406 → 437 (+7.6 %)、pp2048 396 → 429 (+8.3 %)、tg128 10.06 → 10.16 |
| gufo 0.13 DFlash2、カーネル 7.0 → 7.3-perfopt | 59.7 → 59.0 tok/s(変化なし) |
| gufo --sessions 4 -c 65536、同時ストリーム 1 / 2 / 4 | 合計 32 / 50 / 54 tok/s(単一セッションの 59 に対して) — DFlash2 では並列化は報われない |
| --sessions 4 -c 131072 | 拒否: "テキスト実行リクエストが状態の容量を超えている" |

#### 備考

同じマシン、同じモデルファイル、サーバーは一度に一つ、GPU 周波数の固定なし、周囲温度。実行間のばらつきは ±5 tok/s であるため、~5 % 未満の差は n = 12 では決定的ではない。

tubo の短いプロンプトでの向上の約半分は draft flags によるもので、オリジナルの gufo もこれを受け付ける。tubo 独自のエンジンの向上は短いコンテキストでは ~4 % だが、コンテキスト長とともに増える(turbo6 KV はトークンあたりの KV 読み取りを半分にする)。

公開されている tubo は gufo の Nix ツールチェーンを通してはビルドできず(Nix のソースフィルタが削除する CMake 参照 tools/qwen27b/\*.hip のため)、ROCm 7.2 を必要とする。

原文(フランス語)は英語版 [INDEPENDENT_EVO-X2_2026-10-28.md](INDEPENDENT_EVO-X2_2026-10-28.md) の "## Original (French)" を参照。
