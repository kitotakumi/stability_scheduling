# リポジトリ比較レポート：CEGA ／ stability_scheduling

**作成日**：2026-07-21
**対象**：
- **CEGA**（`y-r7/CEGA`, private）
- **stability_scheduling**（予測リアクティブJSSP再スケジューリング研究コード）

**目的**：同一研究室・近接テーマの2つのリポジトリを、研究内容・開発スタイルの両面から客観的に比較する。優劣の断定ではなく、**それぞれの設計思想と現在地の違い**、および**相互に取り込める点**を整理することを狙いとする。

> 注記：本レポートは公開情報とリポジトリ内コードの読解に基づく。CEGAの正式な研究主張・命名意図は原稿等が未整備のため、一部は実装からの推定であることを都度明記する。

---

## 0. 一言まとめ（TL;DR）

| | CEGA | stability_scheduling |
|---|---|---|
| ひとことで | **「基盤の作り込みが先行」**：クリーンなアーキテクチャと現代的ツーリングで再構築された実験プラットフォーム | **「研究ナラティブが先行」**：仮説・統制比較・統計・論文まで一気通貫した完成度の高い研究 |
| 現在地 | エンジニアリング成熟／研究物語はこれから | 研究物語が完成／コード構造はレガシー寄り |

両者は**きれいに補完関係**にある。CEGAは「土台の設計」で、stability_schedulingは「研究の方法論と成果物」で、それぞれ相手が学べるものを持っている。

---

## 1. サマリ比較表

| 観点 | CEGA | stability_scheduling |
|---|---|---|
| **対象問題** | Flexible JSP（FJSP, Brandimarte `mk01/02/06`） | 古典的 JSP（OR-Library `mt/la/ta` 系6問題） |
| **目的関数** | **単一目的**（メイクスパンのみ, `fitness=1/makespan`） | **多目的**（メイクスパン＋安定性＝順列偏差 β=0） |
| **手法数** | **1手法**（OS/MS二重集団の協調GA）＋パラメータ条件比較 | **7手法**の統制比較（GA / ILS×3 / Memetic×3） |
| **手法の核** | OS集団×MS集団の協調共進化・評価戦略が選択式 | 軌道探索(ILS) vs 集団探索(Memetic)＋安定性誘導演算子(PR/repair) |
| **動的/リアクティブ** | 骨組みあり（機械遅延のみ・一部未実装） | 予測リアクティブが研究の中核・完成 |
| **評価・統計** | 記述統計（best/ave/std, BKSギャップ, エントロピー） | 推測統計（HV, Wilcoxon, Cliff's δ, Friedman） |
| **アーキテクチャ** | レイヤード＋イベント駆動（`domain/events/reactive/ga/log`, 26モジュール平均~125行） | フラット・モノリス（ルート直下9ファイル, 最大 `ils_scheduling.py` 1,621行） |
| **依存管理** | **uv + lockfile + ruff**（再現性担保） | pip直叩き（lockfile無し）＋ deap |
| **実験の回し方** | **config駆動**（YAML `sweep`/`fixed`, 直積/zip展開） | スクリプト駆動（`run_v3.py`／`analyze_v3.py`, 1file=1run で resume可） |
| **テスト** | 無し（`__main__`スモークのみ） | 最小限（`test_taillard_rerun.py` 1本のスモーク） |
| **ドキュメント** | README 1本（環境構築・運用中心） | md 21本＋APIEMS2026フルペーパー原稿＋理論・ゼミ資料 |
| **投稿論文** | 無し（CEGA単体には） | **あり**（APIEMS 2026 釜山, フルペーパー原稿完成） |
| **コミット** | 86 commits（2026-04〜07, 約3か月, 単一スカッシュで公開） | 82 commits（2025-10〜, 約9か月） |
| **成熟段階** | 初期〜中期の基盤（engineering-strong / research-thin） | 完成期の研究（paper-ready） |

---

## 2. 研究内容の比較

### 2.1 対象問題：FJSP vs 古典JSP

- **CEGA** は **Flexible JSP** を扱う。各作業が複数の機械候補を持ち、「どの機械に割り当てるか（MS）」と「どの順で処理するか（OS）」を同時に決める、古典JSPより一段広い問題クラス。ベンチマークは Brandimarte（`mk01/02/06`）、BKS（Best Known Solution）との gap で品質を測る。
- **stability_scheduling** は **古典JSP**（機械割当固定）を扱う。その代わり、外乱前スケジュール $S_p$ からの**変更量の小ささ（安定性）**という第二目的を導入し、問題の性格を「大域最適化」から「$S_p$近傍の充填」へとシフトさせている。

→ **問題クラスの広さでは CEGA が上（FJSP）**、**目的の多次元性では stability_scheduling が上（多目的）**。片方が難しくもう片方が易しい、ではなく**難しさの方向が違う**。

### 2.2 目的関数：単一 vs 多目的

- CEGA は現状 **メイクスパン単一目的**（`ga/decoder.py`: `fitness = 1.0/makespan`）。安定性・納期・重みの項はコードに存在しない。ただし `domain/problem.py` に `# 他の属性があればここに追加(例: due_date, weightなど)` というプレースホルダがあり、**多目的化を将来的に想定している構造**にはなっている。
- stability_scheduling は **効率×安定の2目的**を $[0,1]^2$ に正規化し、重み10点掃引でトレードオフ全体を評価。手法比較は主に **union UEA の Hypervolume** で行う。

→ 現時点では **多目的最適化の設計・評価は stability_scheduling が明確に先行**。CEGAが将来ここへ踏み込むなら、stability_scheduling の目的設計・HV評価がそのまま参考資料になる。

### 2.3 手法の核

- **CEGA の distinctive idea**：**OS集団とMS集団を別々に進化させ、互いをパートナーとして協調評価する**（`evaluation_mode: linked/best/random/elite/roulette` で組み合わせ戦略を切替）。これは事実上の**協調共進化GA（Co-Evolutionary GA＝"CEGA" の由来と推定）**。OS交叉は PMX/OX/CX、MS交叉は one/two-point/uniform、選択は roulette/tournament/rank と、**演算子を config で自由に差し替えられる汎用GA基盤**として作られている。
  - ※「CEGA」の正式な定義はリポジトリ内に明記が無く、上記は実装からの推定。命名意図は本人確認が確実。
- **stability_scheduling の distinctive idea**：**同一の安定性誘導演算子（PR / repair）でも、埋め込まれる探索構造（軌道 ILS vs 集団 Memetic）によって効果が非対称に現れる**という構造分析。N5近傍を全手法で揃えた統制条件で、7手法を横断比較する。

→ CEGAは「**汎用で拡張しやすい1つの強いGAエンジン**」、stability_schedulingは「**複数手法の統制比較による構造的発見**」。志向が根本的に異なる（プラットフォーム志向 vs 仮説検証志向）。

### 2.4 動的／リアクティブスケジューリング

- **CEGA**：`reactive/emulator.py`・`experiment/simulation.py`・`ga/scheduler.py` にローリング再スケジューリングの**骨組みが存在**。ただし現状はイベント源が `MachineDelayEvent(time=10, machine=1, delay×3)` のハードコード1件のみで、`JobArrival` は明示的に未実装（`raise`）、機械故障/復旧はスタブ。**足場は組んだが本格運用はこれから**の段階。
- **stability_scheduling**：予測リアクティブ再スケジューリングそのものが**研究の中核として完成**。外乱（作業遅延）シナリオを `scenarios/*.json` として複数規模（`la36_small/middle/large`, `ta21_high`）で用意し、凍結スコープの扱い（`reschedule_time`）まで検証済み。

→ **動的化の完成度は stability_scheduling が大きく先行**。逆にCEGAは、イベント駆動の土台がある分、**将来の動的化を実装しやすい素地**を持っている。

### 2.5 評価・統計の厳密さ

- **CEGA**：**記述統計**中心。run毎に best/ave/std makespan・elapsed_time、世代毎に makespan統計＋**正規化Shannonエントロピー**（OS/MS の多様性）、セッションで BKSギャップ。図は boxplot・条件別 grouped boxplot・推移線・**エントロピーのヒートマップ**。→ **多様性の可視化計装が非常に充実**しているのが特徴。有意差検定・HV は無し（単一目的のため scipy も依存に無し）。
- **stability_scheduling**：**推測統計**中心。HV・Wilcoxon符号順位検定・Cliff's δ・Friedman平均順位まで用い、閾値感度スイープ（P25–P75）でロバスト性まで確認。

→ **統計的検証の厳密さは stability_scheduling**、**探索ダイナミクス（多様性）の計装は CEGA**、と得意分野が分かれる。

---

## 3. 開発スタイル／リポジトリスタイルの比較

ここが両者の個性が最も鮮明に出る部分。

### 3.1 アーキテクチャ

- **CEGA：レイヤード＋イベント駆動の教科書的設計**
  - `domain/`（`@dataclass(frozen=True)` の不変モデル）／`events/`（Observerパターン, `EventEmitter`＋listener protocol）／`ga/`（状態機械 `Scheduler`, factory `create()`）／`reactive/`（Emulator）／`experiment/`（Runner＋Simulation）／`log/`（record factory＋StorageManager＋analyzer）と**関心の分離が明確**。
  - 26モジュール・平均~125行の**小さく分割されたモジュール群**。ABC・factory・不変性・防御的 assert/raise を一貫使用。
- **stability_scheduling：フラット・モノリス**
  - ルート直下に9つの `.py` を並べる構成。`ils_scheduling.py` は **1,621行**の巨大モジュール、`memetic_scheduling.py` も475行。パッケージ階層は持たない。
  - 研究の歴史的経緯（卒論→修論の継続的発展）がそのまま積み上がった形で、**動くことを最優先した実利的構造**。

→ **保守性・拡張性・可読性の設計品質は CEGA が明確に上**。一方 stability_scheduling は、その巨大モジュールの中に**7手法を統制条件で共存させる評価インフラ**（`evaluation.py` による公平比較の担保）を実装しきっており、「汚いが目的は完遂している」実装。

### 3.2 依存管理・ツーリング

- **CEGA**：**uv による現代的な依存管理**。`pyproject.toml`＋`uv.lock`＋`.python-version` でバージョンを完全固定し、`uv sync` でどのPCでも同一環境を再生成。**ruff** を `E,F,I,UP,B,C4,SIM` の充実ルールセットで導入。READMEに依存管理の運用ルール（`uv add/remove` 後は lockfile をセットでコミット等）まで明文化。→ **再現性への配慮が徹底**。
- **stability_scheduling**：README で `pip install numpy scipy matplotlib deap` を直叩き。**requirements.txt / pyproject.toml / lockfile いずれも無し**、linter/formatter設定も無し。ただし `norm_params` のシード固定など、**実験再現性は別の形（乱数決定化）で担保**している。

→ **環境再現性・コード衛生のツーリングは CEGA が明確に上**。stability_scheduling が今すぐ取り込める最も具体的な改善点。

### 3.3 実験の回し方

- **CEGA：config駆動**。YAMLに `fixed`（固定パラメータ木）と `sweep`（`cartesian`＝直積 / `zip`＝ペア変化）を書き、`ExperimentRunner` が展開して `ProcessPoolExecutor` で並列実行。**コードを触らずYAMLだけでパラメータ空間を掃引**できる。掃引例：`pop_size:[100..500]`、crossover_rate、mutation_rate。`num_runs` 最大100、seed=`seed_base+run_id`。
- **stability_scheduling：スクリプト駆動**。`run_v3.py`／`analyze_v3.py` を分離し、`(problem×weight×method×trial)` をフラット化して並列実行。**1ファイル=1run** 保存で途中停止・再開・部分実行に対応（同 `--output-dir` で resume）。増分キャッシュで変更分だけ再計算。

→ **どちらも並列＋再現性を意識した良い設計**。config駆動（CEGA）は宣言的で掃引しやすく、スクリプト駆動＋1file=1run（stability）は resume と部分再計算に強い。**思想は違うが両者とも実験基盤としては成熟**している。

### 3.4 テスト・検証

- 両者とも**自動テストは実質的に無い**（CEGAは `__main__` スモーク、stabilityは `test_taillard_rerun.py` 1本のスモーク）。CEGAは代わりに**不変条件 assert / 明示的 raise** を随所に置いて堅牢性を担保。
- → **この点は両者共通の弱み**。研究コードとしては典型的だが、`pytest` 導入余地は双方にある。

### 3.5 ドキュメント

- **CEGA**：**README 1本**のみ。内容は uv/benchmarks/実行手順/トラブルシュートと**運用ドキュメントとして非常に良質**だが、研究文書・理論ノート・設計ドキュメント・API docstring は無い。コメントは豊富だが実装レベルの日本語。
- **stability_scheduling**：**md 21本**。APIEMS2026フルペーパー原稿（Abstract〜References〜付録作業メモまで）、母艦 `research_document.md`（634行）、理論ノート（`doc/theory/`）、ゼミ記録（`doc/seminar/` 7本）、実験検証md群。→ **研究ナラティブ・意思決定の記録が圧倒的に厚い**。

→ **ドキュメント depth は stability_scheduling が圧倒**。CEGAが研究成果として発表段階に入るなら、ここは最も差が大きい領域。

### 3.6 コミット履歴・粒度

- **CEGA**：約3か月（2026-04-20〜07-19）で86コミット。公開クローンは単一スカッシュコミットで、履歴からの進化追跡はしにくい。
- **stability_scheduling**：約9か月（2025-10〜）で82コミット。コミットメッセージが `doc(APIEMS2026): …` `diag(core_v3): …` のように**種別プレフィックス＋日本語で意思決定を記録**しており、履歴が研究ログとして機能している。

---

## 4. それぞれの強み（相互に学べる点）

### CEGA が優れている点（stability_scheduling が学べる）
1. **クリーンなアーキテクチャ**：domain/events/reactive の関心分離、不変データモデル、イベント駆動。モノリスの解体先として理想形。
2. **現代的ツーリング**：uv＋lockfile＋ruff による環境再現性とコード衛生。**最も低コストで移植できる改善**。
3. **config駆動の宣言的な実験**：YAMLだけで掃引空間を定義できる展開エンジン。
4. **多様性の可視化計装**：世代×スロットのエントロピーヒートマップは探索ダイナミクスの理解に強力。
5. **FJSPという広い問題クラス**への対応（OS/MS二重表現）。

### stability_scheduling が優れている点（CEGA が学べる）
1. **明確な研究仮説と統制比較**：N5近傍を揃えた7手法比較という「何を主張し、何で確かめるか」の設計。
2. **推測統計の厳密さ**：HV／Wilcoxon／Cliff's δ／Friedman＋感度スイープによるロバスト性検証。
3. **多目的最適化と安定性目的の設計**：CEGAが将来多目的化する際の直接の設計図。
4. **予測リアクティブ再スケジューリングの完成した実装**：CEGAの reactive 骨組みの「完成形」の参照。
5. **厚いドキュメントと意思決定ログ**：論文・理論・ゼミ・実験検証の一気通貫。研究の再現性・説明可能性を担保。

---

## 5. 総括：補完関係にある2つのプロジェクト

両者は同じ「GAベースのジョブショップ・スケジューリング」という土俵にありながら、**進化の軸が直交している**：

- **CEGA** は *how to build* を突き詰めた——保守可能で拡張しやすく、再現性の高い**プラットフォーム**。研究の物語（仮説・多目的・統計・論文）はこれから乗せる段階。
- **stability_scheduling** は *what to claim* を突き詰めた——仮説・統制比較・統計・論文まで完成した**研究成果**。コード基盤はレガシー寄りで、CEGAのような近代化余地がある。

もし2つを融合させるなら、理想は明快：**CEGAのアーキテクチャ・ツーリング・config駆動基盤の上に、stability_schedulingの多目的設計・統制比較方法論・統計評価・ドキュメント文化を載せる**こと。実際CEGAの `domain/problem.py` の多目的プレースホルダと reactive 骨組みは、まさにその融合を見越した設計にも読める。

一方が他方より「優れている」のではなく、**研究プロジェクトのライフサイクルの異なる局面を、異なる強みで体現している**——というのが最も公平な結論である。

---

*本レポートはコード・設定・README・コミット履歴の客観的読解に基づく。CEGA側の研究主張・命名意図・今後の計画については、原稿等が未整備の段階のため本人への確認が確実。*
