# Path Relinking を本研究へ適用するための整理メモ
※ 研究メモ + 実装設計書ドラフト  
※ 「direct swap」は一般的なPRの標準用語というより、この研究メモ上の呼称として整理しています。

---

## 1. ざっくり歴史：Path Relinking とは何か

- Path Relinking（PR）は、1990年代に Fred Glover によって提示されたメタヒューリスティクスで、**2つの良い解の間の軌道を系統的にたどり、中間解からさらに良い解を見つける**発想に基づく手法である。[https://www.sciencedirect.com/science/article/pii/S0305054821001441]
- 文脈としては、**Tabu Search / Scatter Search の intensification（集中化）と diversification（多様化）を橋渡しする手法**として発展してきた。[https://www.sciencedirect.com/science/article/pii/S0305054821001441][https://leeds-faculty.colorado.edu/glover/SS-PR%20Template.pdf]
- PR の基本思想は、**高品質解どうしは重要な属性を共有していることが多く、その間の経路にも有望解が存在しやすい**というものである。[https://www.sciencedirect.com/science/article/pii/S0305054821001441]

### PR の基本動作
- 始点解 `s` から目標解 `t` に向かって、
- `t` が持つ属性を `s` に少しずつ導入し、
- 各ステップで「目標解に近づく候補ムーブ」を作り、
- その中から次の中間解を選ぶ、という流れで進む。[https://www.sciencedirect.com/science/article/pii/S0305054821001441]

### 研究上の位置づけ
- 単独法として使うだけでなく、PR は **GA, Memetic Algorithm, GRASP, ILS, Tabu Search, Ant Colony** などとハイブリッド化されてきた。[https://www.sciencedirect.com/science/article/pii/S0305054821001441]
- したがって PR は、「主役の探索法」でもあり得るが、実務上はむしろ**既存探索法の補強器（intensification operator）**として使われることが多い。[https://www.sciencedirect.com/science/article/pii/S0305054821001441]

---

## 2. スケジューリング問題での適用

PR はスケジューリングでも広く使われており、特に**順列表現・機械順序表現**を持つ問題と相性がよい。

### 代表的な適用先
- **Job Shop Scheduling Problem (JSP)**  
  PR を組み込んだ手法として、GRASP+PR や TS/PR が報告されている。[https://mauricio.resende.info/doc/pargjss.pdf][https://arxiv.org/abs/1402.5613]
- **Flexible Job Shop Scheduling Problem (FJSP)**  
  2024年のレビューでも、**Scatter Search + PR** が代表的手法の一つとして位置づけられている。[https://www.sciencedirect.com/science/article/pii/S037722172300382X]
- **Multi-objective Flexible Job Shop Scheduling**  
  PR は Tabu Search と組み合わせた形でも使われている。[https://www.sciencedirect.com/science/article/pii/S0305054821001441]
- **Bi-objective flowshop** など、他のスケジューリング系問題でも PR の利用例がある。[https://www.sciencedirect.com/science/article/pii/S0305054821001441]

### JSP における代表例
- Peng, Lü, Cheng の **TS/PR for JSP** は、ベンチマークに対して高い競争力を示し、**205インスタンス中49個で上界改善、さらに20年以上未解決だった1例を解いた**と報告している。[https://arxiv.org/abs/1402.5613]

### スケジューリングで PR が向いている理由
- 解が「ジョブ順序」「機械ごとの処理順序」のような**属性集合**として自然に表せる。
- 良い解どうしが部分的に似た順序構造を持つことが多く、**その共通構造を壊しすぎずに探索できる**。[https://www.sciencedirect.com/science/article/pii/S0305054821001441]

---

## 3. JSP での PR 方法論の整理

## 3.1 何を「属性」とみなすか
JSP/FJSP では、PR の「属性」は普通は以下のいずれかになる。

- 機械ごとのジョブ列
- オペレーションの相対順序
- 割付 + 順序（FJSP の場合）

あなたの研究では、**機械ごとの作業順序**を解表現にしているため、PR の属性もここに置くのが自然。

---

## 3.2 ムーブの作り方

PR では「目標解へ近づくムーブ」をどう作るかが設計の中心になる。

### (A) direct swap 型
**定義（本メモでの呼称）**  
目標解で位置 `i` にあるべきジョブ `j*` を、現在解のその機械列の中から探し、**その位置まで一気に swap で持ってくる**。

- これは、順列表現に対して「目標解との不一致位置を1つ直す swap」を選ぶ方式とみなせる。
- PR の一般論としても、順列表現では「現在解を目標解に近づける swap」をムーブとして定義するのは自然である。[https://www.sciencedirect.com/science/article/pii/S0305054821001441]

**長所**
- 1回で「ある位置を目標解と一致」させられる。
- 距離減少が明確。
- 実装が単純。
- 反応時間が厳しい再スケジューリングに向く。

**短所**
- 1回の変更がやや大きい。
- JSP では swap 後の順序が**閉路を作る可能性**があるため、N5 と違って feasibility check が必要。

---

### (B) adjacent swap 型
目標ジョブを**隣接 swap を繰り返して**寄せていく方式。

**長所**
- 1手あたりの変化が小さい。
- 安定性破壊を細かく制御しやすい。

**短所**
- パスが長くなる。
- 候補評価回数が増えやすい。
- 即応性が必要な再スケジューリングでは重くなりやすい。

---

### (C) insertion 型
あるジョブを抜き取り、別位置に挿入して目標解へ近づける方式。  
multi-objective PR の整理論文でも、**first insertion / last insertion** は problem-based heuristic として整理されている。[https://www.sciencedirect.com/science/article/pii/S0305054821001441]

**長所**
- 順列表現では強力なことが多い。
- swap より少ない手数で目標順序へ近づける場合がある。

**短所**
- 1回の move で複数の相対順序を同時に変える。
- あなたの「順位変更量ベース」の安定性指標との対応が、direct swap よりやや読みづらい。

---

### (D) shift 型
要素をずらして目標解へ近づける方式。  
これも permutation 向けの problem-based heuristic として整理されている。[https://www.sciencedirect.com/science/article/pii/S0305054821001441]

---

## 3.3 候補の選び方

PR のもう1つの設計軸は、「作った候補の中から何を選ぶか」。

### 文献上の代表分類
- **ランダム選択**[https://www.sciencedirect.com/science/article/pii/S0305054821001441]
- **単一目的で最良選択（pure）**[https://www.sciencedirect.com/science/article/pii/S0305054821001441]
- **重み付き和（aggregation combined）**[https://www.sciencedirect.com/science/article/pii/S0305054821001441]
- **Pareto 非劣候補から選択**[https://www.sciencedirect.com/science/article/pii/S0305054821001441]
- **decomposition-based**[https://www.sciencedirect.com/science/article/pii/S0305054821001441]

### 本研究との対応
あなたの研究はすでに
- 安定性 `D`
- メイクスパン `MS`
- 重み `λ`
- min-max 正規化  
を使う設計になっているので、**PR の候補選択も aggregation combined（重み付き和）で統一するのが最も自然**。

---

## 4. 本研究での適用方針

以下は、あなたの現在の研究（再スケジューリング、安定性重視、ILS ベース）に合わせた、最も自然な PR の入り方。

## 4.1 PR をどこに入れるか
現状の構成は
- 局所探索：N5 山登り
- 摂動：insert / swap
- 目的：安定性 `D` と `MS` の重み付き和
- 将来的に「初期解へ戻す」操作として PR を検討  
という流れだった。

したがって PR の役割は、**新しい大域探索器ではなく、ILS 内の“制御された摂動 / 回帰操作”**として入れるのがよい。

### 推奨する位置づけ
- `局所最適解 x` に到達
- `x` が元スケジュールから離れすぎたとき、
- `初期スケジュール S_p` を guiding solution にして PR をかける
- 経路上で見つかった最良中間解を次の探索の出発点にする

つまり、

> **「壊しすぎた構造を、全部戻すのではなく、良い点だけ残して少し戻す」**

という使い方。

---

## 4.2 direct swap 採用理由

本研究の初版 PR では、**direct swap を第一候補**にするのが合理的。

### 理由1：安定性指標と整合的
あなたの安定性は、**ジョブの投入順序・順位変更量**を直接見ている。  
direct swap は「目標解のある位置を一致させる」操作なので、**順位差の縮小と move の意味が対応しやすい**。

### 理由2：実装が最も明快
各機械について
- 不一致位置を見つける
- そこに来るべきジョブを探す
- swap する  
だけで候補が作れる。  
AI に実装させるときも誤解が少ない。

### 理由3：候補数が制御しやすい
各ステップの候補数は基本的に  
**「全機械の不一致位置数の総和」**  
になるので、adjacent swap より膨れにくい。

### 理由4：再スケジューリングの即応性に合う
あなたの計算機実験では、GA より ILS の方が短時間で良い結果を出している。  
その流れでは、PR も**細かすぎる adjacent swap より、短い経路を作れる direct swap の方が親和的**。

### 理由5：初期実装として比較しやすい
現状すでに摂動として `insert / swap` を持っているため、  
PR はまず **direct swap で最小実装**し、その後に
- adjacent swap PR
- insertion PR
- hybrid PR  
と比較すればよい。

---

## 4.3 注意点
- N5 は「閉路を作らない」ことが知られているが、**PR の一般 swap はその保証がない**。したがって、**候補ごとに実行可能性チェックが必要**。
- あなた自身のスライドでも、PR を入れる場合は**実行可能解チェックのコスト検証が必要**としていた。この点は本質的な注意点。  
- したがって、本研究における PR は **“安定性に優しいが、評価は重い可能性がある補助操作”** と位置づけるべき。

---

## 5. 本研究向けの具体アルゴリズム案（AI実装用）

## 5.1 役割定義
- **Initiating solution**: 現在の局所最適解 `S_cur`
- **Guiding solution**: 外乱前の初期スケジュール `S_ref = S_p`
- **出力**: PR 経路上で見つかった最良中間解 `S_best`

---

## 5.2 解表現
- 解は**機械ごとのジョブ列**で表す。
- 各機械 `k` について列 `seq[k] = [j_1, j_2, ..., j_n]` を持つ。

---

## 5.3 距離（差分集合）
各機械 `k` に対して、不一致位置集合を

\[
\delta_k(S,T)=\{ i \mid seq_S[k][i] \neq seq_T[k][i] \}
\]

とする。

全体の候補ムーブ集合は、この不一致位置の全組から作る。

---

## 5.4 候補生成（direct swap）
各機械 `k`、各不一致位置 `i ∈ δ_k(S,T)` に対して：

1. 目標解 `T` で位置 `i` にいるべきジョブ `j* = seq_T[k][i]` を取得
2. 現在解 `S` の機械 `k` の列の中で、`j*` がいる位置 `q` を探す
3. `seq_S[k][i]` と `seq_S[k][q]` を swap
4. 得られた解 `S'` を候補に追加

これで、少なくとも位置 `i` は目標解に一致する。

---

## 5.5 候補評価
各候補 `S'` について：

1. **実行可能性チェック**
   - 閉路が出るなら棄却
   - あるいはガントチャート再生成／スケジュールデコードで評価可能なら採用

2. **メイクスパン計算**
   - 既存の評価器（Taillard のアルゴリズム or ガントチャート再生成）を使う

3. **安定性計算**
   - 既存の順位変更量ベース安定性関数 `D(S_p, S')` を使う

4. **総合スコア**
   - あなたの既存目的関数と揃えて  
   \[
   F(S') = \lambda \hat{D}(S_p,S') + (1-\lambda)\hat{MS}(S')
   \]
   を使う  
   （`\hat{D}`, `\hat{MS}` は min-max 正規化後）

5. **次の中間解選択**
   - `F(S')` が最良の候補を次状態に採用

---

## 5.6 終了条件
以下のいずれかで停止：

- 目標解 `S_p` に到達
- 候補が1つも作れない
- 最大ステップ数 `L_max` 到達
- 改善なしが `stall_limit` 回続く
- 時間制限到達

---

## 5.7 戻り値
**最終到達解ではなく、経路上の最良解 `S_best` を返す**のがよい。  
PR は途中で一時的に悪化してもよいが、最終的に使いたいのは「経路上のベスト」である。[https://www.sciencedirect.com/science/article/pii/S0305054821001441]

---

## 5.8 擬似コード

```pseudo
function PATH_RELINKING_DIRECT_SWAP(S_cur, S_ref, lambda, L_max):
    S = S_cur
    S_best = S_cur
    F_best = evaluate_weighted(S_cur, S_ref, lambda)