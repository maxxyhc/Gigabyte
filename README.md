# AORUS MASTER 16 AM6H 規格問答系統

針對 [GIGABYTE AORUS MASTER 16 AM6H](https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp)
產品規格頁的 RAG 問答系統，支援繁體中文、英文與中英混合提問，
在 **4GB VRAM 限制內**執行。

RAG 的核心邏輯（chunking、retrieval、generation）全部手寫，未使用 LangChain 或 LlamaIndex。

## 成果摘要

| 指標 | 結果 | 量測環境 |
|---|---|---|
| VRAM 佔用 | **2,893 MiB** / 4,096 MiB | Tesla T4 |
| 檢索 Recall@3 | **1.000**（23 題可回答，全部命中） | 確定性，跨裝置一致 |
| 答案事實正確率 | **1.000** | T4, temperature=0 |
| 拒答正確率 | **1.000**（7 題不可回答全部婉拒） | T4, temperature=0 |
| 誤拒率 | **0.000** | T4, temperature=0 |
| 捏造次數 | **0** | T4, temperature=0 |
| TTFT p50 / p95 | **0.34s / 0.49s** | T4 |
| 生成速度 | **52.7 tok/s** | T4 |

完整的 T4 執行紀錄（含 `nvidia-smi` 輸出）保存在
[`notebooks/colab_bench.ipynb`](notebooks/colab_bench.ipynb)。

---

## 快速開始

### 環境需求

- Python 3.12（`uv` 會自動安裝）
- [`uv`](https://docs.astral.sh/uv/)
- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) 的 `llama-server`
- 約 3GB 磁碟空間（模型檔）

### 1. 環境

```bash
git clone https://github.com/maxxyhc/Gigabyte.git && cd Gigabyte
uv sync --frozen
```

`--frozen` 會完全照 `uv.lock` 安裝，不重新解析依賴。

### 2. 安裝 llama.cpp

macOS：

```bash
brew install llama.cpp
```

Linux / CUDA（需自行編譯，官方 release 未提供 Linux CUDA 預編譯檔）：

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build llama.cpp/build --target llama-server -j
```

### 3. 下載模型（2.3 GB）

```bash
mkdir -p models && curl -L -o models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
```

### 4. 建立向量索引（CPU，約 10 秒）

```bash
uv run python src/embed.py
```

索引是可重建產物，未進版控。`data/chunks.jsonl` 已包含在 repo 中，
若要從原始網頁重新產生：`uv run python src/fetch.py --force && uv run python src/parse.py`。

### 5. 啟動 llama-server

```bash
llama-server -m models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 --ctx-size 4096 \
  --cache-type-k q8_0 --cache-type-v q8_0 -ngl 99 --jinja
```

這些旗標**就是 4GB 預算本身**，理由見〈模型選擇與 VRAM 預算〉。

> Colab 已佔用 8080 埠，在 Colab 上請改用其他埠並以 `--base-url` 傳給下游程式。

### 6. 提問

單次查詢：

```bash
uv run python src/rag.py "這台的顯卡有多少記憶體？" --sources
```

互動模式：

```bash
uv run python src/rag.py
```

### 7. 重現評測

檢索評測（不需要 llama-server，約 30 秒）：

```bash
uv run python eval/run_eval.py
```

生成評測（需要 llama-server，90 次生成，約 3–5 分鐘）：

```bash
uv run python eval/run_gen_eval.py
```

在 4GB 環境下的完整重現（含 VRAM 舉證）請執行
[`notebooks/colab_bench.ipynb`](notebooks/colab_bench.ipynb)。

---

## 系統架構

```
規格網頁 ──fetch.py──> HTML 快照 ──parse.py──> chunks.jsonl（21 筆）
                                                    │
                                    embed.py（CPU）──┤
                                                    ▼
問題 ──> retrieve.py ──> index.py ──> Dense（numpy）+ BM25（手寫）
                             │              └─ RRF 融合 ─ SKU 路由 ─ top-3
                             ▼
                        rag_prompt.py ──> llm.py ──SSE──> llama-server
                                              │
                                          metrics.py（TTFT / TPS）
```

| 模組 | 職責 |
|---|---|
| `src/fetch.py` | 抓取網頁並落檔，含反爬蟲 header 與內容健全性檢查 |
| `src/parse.py` | HTML → 欄位 K-V → chunks，含三機型合併與比較 chunk |
| `src/embed.py` | CPU 端建立向量索引，輸出 `.npy` |
| `src/index.py` | Dense 檢索（numpy）與手寫 BM25 |
| `src/retrieve.py` | RRF 融合、SKU 路由、設定物件 |
| `src/rag_prompt.py` | Prompt 組裝與 context 格式化 |
| `src/llm.py` | 手寫 SSE 串流客戶端 |
| `src/metrics.py` | TTFT / TPS 的定義 |
| `src/rag.py` | 串接檢索與生成 |

### 手寫了什麼

禁用框架的規範下，以下全部是自行實作：

- **HTML 解析**：BeautifulSoup 選取節點，自行處理 `<br>` 換行、trademark 符號、註腳
- **Chunking**：以規格欄位為單位，自行組裝雙語標頭
- **向量索引**：`numpy` 矩陣，21×1024，檢索即一次矩陣乘法
- **BM25**：tokenizer、IDF、文件長度正規化全部手寫
- **RRF 融合**：不依賴任何檢索函式庫
- **SSE 解析**：逐行解析 `text/event-stream`，不使用 OpenAI SDK

外部依賴僅四項：`requests`、`beautifulsoup4`、`numpy`、`sentence-transformers`。

---

## 資料解析

### 網頁取得

規格頁位於 Akamai 之後，`requests.get` 直接請求會得到 `Access Denied`。
需要完整的瀏覽器 header 組合（`User-Agent`、`Accept-Language`、`Sec-Ch-Ua`、
`Sec-Fetch-*`、`Upgrade-Insecure-Requests`）才能通過，缺任一項即失敗。
頁面本身是 server-rendered，不需要 headless browser。

`fetch.py` 除了檢查 HTTP 狀態碼，另外驗證回應中確實含有
`desktop-spec-content` 標記——反爬蟲層可能回傳 200 加上一個錯誤頁面，
只看狀態碼會誤判成功。

原始 HTML 快照已提交至 `data/raw/`，讓解析步驟能離線重現，
評測時也不會反覆對來源站發送請求。

### 結構化擷取

頁面同時提供行動版與桌面版兩種佈局。**行動版只渲染第一個機型的欄位**，
桌面版才含完整三欄，因此解析目標是 `div.desktop-spec-content`：

```html
<div class="spec-column">
  <div class="multiple-title"><div>顯示晶片</div></div>   <!-- 17 個欄位 -->
</div>
<div class="swiper-wrapper">
  <div class="swiper-slide">                              <!-- 每個機型一欄 -->
    <div class="spec-item-list" data-spec-row="2">…</div>
```

共擷取 **17 個欄位**：作業系統、中央處理器、顯示晶片、顯示器、記憶體、儲存裝置、
鍵盤種類、連接埠、音效、通訊、視訊鏡頭、安全裝置、電池、變壓器、尺寸、重量、顏色。

機型名稱來自頁面副標題，欄位對齊使用 `data-spec-row` 而非陣列位置，
避免某一欄缺少某列時整排位移。

實作上兩個容易忽略的細節：

- **`<br>` 必須保留為換行。** BeautifulSoup 的 `get_text()` 會直接略過 `<br>`，
  導致 `16" 16:10OLED WQXGA…` 這種黏行，會同時污染 embedding 與模型讀到的 context。
- **移除 trademark 符號必須在 NFKC 正規化之前。** NFKC 會把 `™`(U+2122)
  展開成字母 `TM`，`RTX™ 5090` 變成 `RTXTM 5090`，之後再也濾不掉。

---

## Chunking 與檢索設計

### 語料規模決定了設計方向

整份規格僅約 2,400 字元、17 個欄位。這是**精準度問題，不是規模問題**，
因此刻意避開為大規模設計的技術：不使用向量資料庫（21 個向量，
一次 `matrix @ query` 即為全部）、不做二次切分、不做 overlap。

### 一個欄位一個 chunk

規格表本身已經是語意完整的切分單位，再切會破壞 key-value 對應。
每個 chunk 自帶雙語標頭：

```
[產品] GIGABYTE AORUS MASTER 16 AM6H
[機型] 全機型 (BZH / BYH / BXH)
[欄位] 顯示晶片 (Graphics / GPU / 顯卡 / 獨顯 / NVIDIA / RTX / VRAM)
NVIDIA GeForce RTX 5090 Laptop GPU
24GB GDDR7
```

標頭裡的別名表（`data/aliases.json`）是支援中英混合提問的主要機制——
它把兩種語言的詞彙放進 chunk 本文，使 dense 與 lexical 檢索都能命中。
評測顯示這是**影響最大的單一設計**（見下）。

### 三個 SKU 的處理

AM6H 有 BZH / BYH / BXH 三種機型，但**只有顯示晶片欄位有差異**，
其餘 16 個欄位三機型完全相同。

若照機型展開會產生 51 個 chunk，其中大量近似重複——它們會互相瓜分排名，
並把 `k=3` 全部佔滿，使跨欄位問題取不到第二個欄位。

因此 `parse.py` 對每個相異值只產生一個 chunk，並記錄適用機型；
對有差異的欄位額外產生一個**比較 chunk**。最終 21 筆：

| 類型 | 數量 | 說明 |
|---|---|---|
| `spec` | 16 | 三機型共用 |
| `sku` | 3 | 顯示晶片，各機型專屬 |
| `compare` | 1 | 顯示晶片三機型並列 |
| `derived` | 1 | 規格總覽（各欄位首行） |

`retrieve.py` 據此做**互斥路由**：

- 問題提到機型 → 只給該機型的 chunk，排除比較 chunk
- 問題未提機型 → 只給比較 chunk，排除各機型專屬 chunk

這解決了一個實際錯誤：未做路由前，「顯卡多強」會回傳三筆中分數最高的一筆
（差距僅 0.003，實質隨機），使用者會被告知「RTX 5070 Ti」——三張裡最弱的那張，
且語氣像是唯一答案。

### Hybrid 檢索

Dense（bge-m3 cosine）與 BM25 並用，以 RRF 融合。兩者的失效模式互補：

| 問題 | Dense | BM25 |
|---|---|---|
| 「插頭多大顆」 | `spec.adapter` ✅ | 全部 0 分 ❌（語料無此詞） |
| 「microSD 支援哪個 UHS standard」 | `spec.storage` ❌ | `spec.ports` ✅（3.73 分） |

實作上的三個決定：

**融合用 RRF 而非加權分數相加。** cosine 落在 [0,1]，BM25 無上界且隨語料變動，
相加需要正規化，而 min-max 正規化在 21 篇文件上極不穩定。RRF 只用名次，天然免疫。
論文預設 `rrf_k=60` 是為上千篇的排名設計，在 21 篇上會把第 1 名與第 10 名壓得幾乎等價，
因此本專案取 `rrf_k=10`。

**零分的檢索器不參與投票。** BM25 對「插頭多大顆」給所有 chunk 0 分，
若讓它照樣排名，任意的 tie-break 順序會蓋過真的找到答案的 dense 檢索器。

**BM25 的 tokenizer 用 character bigram，不用 jieba。** 少一個依賴，
而且斷詞器會切壞 `AM6H`、`WQXGA`、`Gen4x4` 這類型號字串，而規格問答正是靠這些字串定位。

**IDF 使用 Lucene 變體。** Robertson 原始公式
`ln((N - df + 0.5) / (df + 0.5))` 在詞出現於超過半數文件時會**變成負值**，
使包含該詞的文件反而被扣分。在 21 篇文件、character bigram 的條件下這不是邊緣情況——
實測有 **12 個詞為負**，包含 `am6h`、`byh`、`機型`、`產品`。
改用 `ln(1 + (N - df + 0.5) / (df + 0.5))` 後為 0 個。
原始公式保留在 `IDF_FORMULAS` 中，可用 `--idf-formula robertson` 重現此問題。

---

## 模型選擇與 VRAM 預算

### 選擇

**`Qwen3-4B-Instruct-2507`，Q4_K_M 量化**

| 元件 | 佔用 | 說明 |
|---|---|---|
| 模型權重 Q4_K_M | ~2.4 GB | 檔案 2,497 MB（2.33 GiB） |
| KV cache（4096 ctx, q8_0） | ~0.4 GB | `--cache-type-k/v q8_0` |
| Embedding 模型 | **0** | bge-m3 跑在 CPU |
| **T4 實測總計** | **2,893 MiB** | 餘裕 1,203 MiB |

選擇理由：

**繁體中文能力是硬需求。** Qwen 系列在中文評測上明顯領先同尺寸的
Llama、Gemma、Mistral，而本系統需要處理繁中提問並以繁中作答。
實測中亦發現一個附帶效果：RAG 提供的繁中 context 使模型的簡體字洩漏
從 3 題降為 0 題（見評測），因此未加入簡繁轉換層。

**4B 是預算內能取到的最大尺寸。** Q4_K_M 在 2.4GB 左右，加上量化後的 KV cache
仍有近 30% 餘裕。更大的 7B/8B 模型即使 Q4 也會超過 4GB。

**Embedding 不佔 VRAM 是刻意的架構決定。** 索引是 21 個向量離線建一次存成 `.npy`，
查詢時只需編碼問題本身，在 CPU 上僅數十毫秒、完全無感。
把整個 VRAM 預算留給 LLM，是本專案最划算的一個取捨。

**Fallback：`Qwen3-1.7B-Instruct` Q4_K_M（~1.1 GB）。** 若目標裝置的
VRAM 餘裕更緊（例如同時要跑顯示輸出），可直接替換模型檔，其餘程式無需改動。

### 為什麼用 `llama-server` 而非 Python binding

`llama-server` 提供 OpenAI 相容的 `/v1/chat/completions` 與 SSE 串流，
使 **TTFT 可以在傳輸層量測**——從請求送出到第一個 content token 抵達。
這是交付項目之一，透過 binding 量測會混入 Python 層的開銷。
Server/client 分離也讓評測腳本能獨立於推論行程重複執行。

### 啟動旗標的意義

| 旗標 | 作用 |
|---|---|
| `--ctx-size 4096` | 21 個 chunk 的語料綽綽有餘，同時壓住 KV cache |
| `--cache-type-k/v q8_0` | KV cache 量化，約減半 |
| `-ngl 99` | 全部層 offload 到 GPU |
| `--jinja` | 使用模型內建的 chat template |

### 4GB 舉證

開發在 Apple Silicon 上進行，但 Metal 使用統一記憶體，**無法證明 4GB 限制**。
因此所有 VRAM 與延遲數據取自 Colab Tesla T4：

```
baseline before server :      0 MiB
after model load       :   2887 MiB   (+2887 MiB)
under load             :   2891 MiB   (+2891 MiB over baseline)
after full evaluation  :   2893 MiB

pid, process_name, used_gpu_memory [MiB]
32281, /content/llama.cpp/build/bin/llama-server, 2890 MiB

4096 MiB budget: WITHIN
```

量測方式刻意取**差值**而非絕對值，扣除 runtime 自身的佔用；
並在跑過真實查詢**之後**才讀數，因為 KV cache 隨實際處理的 token 成長，
剛載入時的數字會偏低。

---

## 評測

### 評測集

`eval/golden_set.jsonl`，30 題手寫，語系配比 15 中文 / 10 英文 / 5 中英混合：

| 題型 | 題數 | 測什麼 |
|---|---|---|
| `single_field` | 12 | 單一欄位事實 |
| `cross_field` | 6 | 需要兩個以上欄位才能回答 |
| `model_diff` | 5 | 三機型的異同 |
| `unanswerable` | 7 | 規格頁沒有的資訊，必須婉拒 |

設計上刻意避免的兩件事：

- **不使用欄位原名提問。** 問「顯示晶片是什麼」等於在測字串比對。
  題目使用使用者真實會用的說法（「打電動夠不夠力」「插頭多大顆」「what GPU does it have」），
  其中 8 題的用詞**只存在於別名表**，用來單獨量測別名的效果。
- **拒答不以固定措辭比對。** 婉拒有上百種說法，因此 `expect_refusal` 是獨立的布林指標，
  並以 `must_not_contain` 捕捉捏造（例如出現具體價格）。

字串比對前統一正規化（casefold、去空白、去連字號、`×`→`x`），
且每個事實接受多種表面形式，避免把「24 個核心」判成錯誤答案。

### 檢索結果

不涉及 LLM，完全確定性，在 macOS 與 T4 上結果逐位相同。

```
configuration               R@1    R@3  Hit@3    MRR   alias R@1
----------------------------------------------------------------
dense only, alias         0.471  0.986  1.000  0.826       0.250
bm25 only, alias          0.674  0.891  0.913  0.891       0.750
hybrid, no alias          0.304  0.768  0.826  0.616       0.250
hybrid + alias            0.645  1.000  1.000  0.928       0.750
  ...no SKU routing       0.609  0.986  1.000  0.884       0.750
  ...no overview chunk    0.645  1.000  1.000  0.935       0.750
(R@1 ceiling)             0.775      -      -      -       1.000
```

**R@1 的上限不是 1.0。** 部分問題需要多筆 gold chunk，而第 1 名只有一個位置，
該類問題的 R@1 上限為 `1/|gold|`。全體平均上限為 **0.775**，
因此 R@1 應與此比較，而非與 1.0 比較。

分題型（hybrid + alias）：

| 題型 | n | R@1 | 上限 | R@3 | MRR |
|---|---|---|---|---|---|
| `single_field` | 12 | 0.750 | 1.000 | 1.000 | 0.861 |
| `cross_field` | 6 | 0.500 | **0.500** | 1.000 | 1.000 |
| `model_diff` | 5 | 0.567 | **0.567** | 1.000 | 1.000 |

**分析：**

1. **`hybrid + alias` 的 R@3 為 1.000**——23 題可回答的問題，
   所需的 gold chunk 全部落在前三名內。餵給 LLM 的是 top-3，因此檢索層已無瓶頸。

2. **別名是影響最大的單一設計。** 移除後 R@3 由 1.000 掉到 0.768；
   在專門依賴別名的 8 題子集上，R@1 由 0.750 掉到 0.250（三倍差距）。
   代價是每個 chunk 的標頭變長、稀釋語意重心，但總體效益遠大於成本。

3. **兩種檢索器單獨使用皆不足。** dense 的 R@3 為 0.986、BM25 為 0.891，
   合併後才達 1.000。值得注意的是 BM25 單獨的 R@1（0.674）**高於**合併後（0.645）——
   BM25 命中時往往直接排第一，但落空時完全找不到；合併犧牲了些許首位命中率，
   換取「不再整題落空」。以 top-3 餵給模型的用法而言，這個交換是划算的。

4. **跨欄位與機型比較兩類已達理論上限**，無進步空間。
   唯一有餘裕的是單欄位題的首位排序，3 題排在第 2–3 名，
   失誤集中在語意高度相近的欄位（顯卡 MHz vs 螢幕 Hz、microSD vs 儲存、network vs 網路孔）。

5. **規格總覽 chunk 目前未展現效益**（移除後 MRR 反而由 0.928 微升至 0.935）。
   但此結論不成立——評測集中**沒有一題是「這台有什麼特色」這類籠統問題**，
   而那正是該 chunk 存在的理由。屬於評測覆蓋不足，非設計失敗。

### 生成結果

Tesla T4，`temperature=0`，90 次生成（30 題 × 3 設定）。

```
config      facts   part  refuse falseR  fab   簡  TTFT50  TTFT95    TPS
------------------------------------------------------------------------
no RAG      0.174  0.337   0.143  0.043    1    3   0.06s   0.08s   56.8
RAG k=1     0.696  0.819   0.857  0.435    0    1   0.08s   0.17s   52.7
RAG k=3     1.000  1.000   1.000  0.000    0    0   0.34s   0.49s   52.7
```

**分析：**

1. **`RAG k=3` 五項品質指標全部滿分**：事實正確率 1.000、
   7 題不可回答全部婉拒、可回答題零誤拒、零捏造、零簡體字。

2. **無 RAG 的失效模式不是「答不出來」，是「有自信地編造」。**
   7 題不可回答的問題僅 1 題承認不知道。實際輸出：

   | 問題 | 模型回答 | 規格頁事實 |
   |---|---|---|
   | 電池容量 | 「92.5Wh」 | **99Wh** |
   | 最多支援多少記憶體 | 「128 GB，DDR5 5600 CL30」 | **Up to 64GB**（CL30 為虛構） |
   | 台灣官方售價 | 「新台幣 19,999 元」 | 頁面無價格資訊 |
   | 原廠保固 | 「三年」 | 頁面無保固年限 |

   這是本系統存在的理由：AM6H 發表於模型訓練資料之後，
   缺乏 grounding 時模型會以合理的語氣填補空白，並附上
   「建議至官方網站查詢最新資訊」這類使人更易採信的措辭。

3. **`k=1` 的主要失效是誤拒（0.435），而非答錯。**
   只提供一筆 context 時經常不是所需的欄位，模型遵守 grounding 規則回答
   「規格頁未提供」。捏造次數仍為 0，顯示 prompt 的約束有效——
   問題出在檢索給的資料不足。這是 `k=3` 的直接依據。

4. **RAG 附帶降低了簡體字洩漏（3 → 1 → 0）。**
   context 中的繁中規格原文提供了可引用的字形，模型不需自行生成。
   原本考慮引入 OpenCC 轉換層，此結果使該依賴變得不必要。

### 延遲分析

| | TTFT p50 | TTFT p95 | TPS |
|---|---|---|---|
| 無 RAG（prompt 中位數 78 token） | 0.06s | 0.08s | 56.8 |
| RAG k=1（prompt 中位數 517 token） | 0.08s | 0.17s | 52.7 |
| RAG k=3（prompt 中位數 906 token） | 0.34s | 0.49s | 52.7 |

**TTFT 幾乎完全由 prompt 長度決定，而非模型的生成能力。**
context 由 78 token 增至 906 token，首字延遲上升約 5.7 倍，
但生成速度僅下降 7%（56.8 → 52.7 tok/s）。

實務含意：若要壓低首字延遲，有效的手段是**縮短 context**
（精簡欄位內容、降低 `k`、壓縮 chunk 標頭），換用更快的模型幫助有限。
以本專案的取捨而言，`k=3` 比 `k=1` 多花約 0.26 秒，
換來誤拒率由 0.435 降為 0——在規格查詢的場景中，答不出來的代價遠高於半秒延遲。

**平台比較**（同樣 `temperature=0`）：

| | Apple M2 Pro (Metal) | Tesla T4 (CUDA) |
|---|---|---|
| TTFT p50 | 0.77s | **0.34s** |
| TTFT p95 | 1.51s | **0.49s** |
| TPS | 42.1 | **52.7** |

T4 在兩項指標上皆優於 M2 Pro。

### 量測方法的限制

以下三點為誠實揭露，避免對數據做過度解讀：

**`temperature=0` 在同一裝置上可重現，跨裝置不可重現。**
無 RAG 的事實正確率在 M2 Pro 為 0.261、在 T4 為 0.174。
貪婪解碼在不同硬體上的浮點運算順序不同，會走出不同路徑。
但 `RAG k=3` 在兩個平台上皆為 1.000，主要結論不受影響。

**拒答的判定使用啟發式關鍵詞。** 婉拒沒有固定措辭，
本系統以常見說法的關鍵詞集合判斷，並輔以 `must_not_contain` 捕捉捏造。
此為比率指標，不作為單題的通過與否判準。

**檢索分數無法作為拒答訊號。** 曾嘗試以「最高分過低即婉拒」實作提前拒答，
但可回答題與不可回答題的平均首位分數為 0.1721 vs 0.1528，差距不足以劃分閾值。
原因是 RRF 基於名次而非相似度，完全不相關的文件同樣會取得名次。
拒答因此完全交由生成層處理，而該層達成 1.000。

---

## 專案結構

```
├── AGENTS.md                     設計決策、硬性限制、程式碼紀律
├── pyproject.toml / uv.lock      uv 環境
├── data/
│   ├── raw/spec_zh.html          原始網頁快照（可重現性基準）
│   ├── chunks.jsonl              21 筆檢索單元
│   └── aliases.json              17 欄位的中英別名表
├── src/                          9 支模組，見〈系統架構〉
├── eval/
│   ├── golden_set.jsonl          30 題評測集
│   ├── run_eval.py               檢索評測
│   ├── run_gen_eval.py           生成評測
│   └── results/gen_eval.json     每題的完整答案與計時
└── notebooks/colab_bench.ipynb   T4 完整重現（含執行輸出）
```

未進版控者：`models/`（模型檔）、`data/index/`（向量索引）、`.venv/`，
皆可由上述指令重建。

## 已知限制與後續方向

- **評測集缺少籠統問題**，導致規格總覽 chunk 的效益無法量測。補 2–3 題後重跑即可判定去留。
- **單欄位題的首位排序仍有 3 題落在第 2–3 名。** 已定位一個可行改進：
  將出現在全部 21 筆文件中的詞從 BM25 移除（例如產品名 `am6h`），
  這類詞不具區分能力卻會擾動排名。
- **僅涵蓋單一機種。** 擴充至產品線時，現行的「一個相異值一個 chunk + 適用機型 metadata」
  設計可直接沿用，`parse.py` 已依此實作而非針對顯示晶片特例化。
- **拒答判定可再嚴謹。** 目前為關鍵詞啟發式，改用 LLM-as-judge 可提高一致性，
  代價是評測本身需要額外的推論成本。
