# AORUS MASTER 16 AM6H 規格問答系統

針對 [GIGABYTE AORUS MASTER 16 AM6H](https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp)
產品規格頁的 RAG 問答系統，支援繁體中文、英文與中英混合提問，
在 **4GB VRAM 限制內**執行。

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

### 4. 建立向量索引

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

檢索評測：

```bash
uv run python eval/run_eval.py
```

生成評測：

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

以下元件均為自行實作：

- **HTML 解析**：BeautifulSoup 選取節點，自行處理 `<br>` 換行、trademark 符號、註腳
- **Chunking**：以規格欄位為單位，自行組裝雙語標頭
- **向量索引**：`numpy` 矩陣，21×1024，檢索即一次矩陣乘法
- **BM25**：tokenizer、IDF、文件長度正規化全部手寫
- **RRF 融合**：只用兩個檢索器的名次，不用分數
- **SSE 解析**：逐行解析 `text/event-stream`，含 UTF-8 解碼與 usage 擷取

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

**4B 是預算內能取到的最大尺寸。** Q4_K_M 在 2.4GB 左右，加上量化後的 KV cache
仍有近 30% 餘裕。更大的 7B/8B 模型即使 Q4 也會超過 4GB。

**Embedding 不佔 VRAM。** 索引是 21 個向量離線建一次存成 `.npy`，
查詢時只需編碼問題本身，在 CPU 上僅數十毫秒。
把整個 VRAM 預算留給 LLM。

**啟動旗標本身就是預算的一部分，不是預設值。** `--ctx-size 4096` 對 21 個 chunk
的語料綽綽有餘，同時把 KV cache 限制在小尺寸；`--cache-type-k q8_0 --cache-type-v q8_0`
將 KV cache 量化，佔用約減半——省下這一步，同樣的 context 長度會多佔數百 MB，餘裕明顯縮水；
`-ngl 99` 將全部層 offload 至 GPU，確保量到的就是完整的 GPU 佔用而非部分留在 CPU 的假象；
`--jinja` 使用模型內建的 chat template。

**Fallback：`Qwen3-1.7B-Instruct` Q4_K_M（~1.1 GB）。** 若目標裝置的
VRAM 餘裕更緊（例如同時要跑顯示輸出），可直接替換模型檔，其餘程式無需改動。

### 為什麼用 `llama-server` 而非 Python binding

`llama-server` 提供 OpenAI 相容的 `/v1/chat/completions` 與 SSE 串流，
使 **TTFT 可以在傳輸層量測**——從請求送出到第一個 content token 抵達；
透過 binding 量測會混入 Python 層的開銷。
Server/client 分離也讓評測腳本能獨立於推論行程重複執行。

### 4GB 舉證

開發在 MacBook 上進行，但 Metal 使用統一記憶體，**無法證明 4GB 限制**。
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

兩項設計選擇：

- **不使用欄位原名提問。** 問「顯示晶片是什麼」等於在測字串比對。
  題目使用使用者真實會用的說法（「打電動夠不夠力」「插頭多大顆」「what GPU does it have」），
  其中 8 題的用詞**只存在於別名表**，用來單獨量測別名的效果。
- **拒答不以固定措辭比對。** 婉拒有上百種說法，因此 `expect_refusal` 是獨立的布林指標，
  並以 `must_not_contain` 捕捉捏造（例如出現具體價格）。

字串比對前統一正規化（casefold、去空白、去連字號、`×`→`x`），
且每個事實接受多種表面形式，避免把「24 個核心」判成錯誤答案。

### 受測設定

| 名稱 | 說明 |
|---|---|
| dense | bge-m3 向量檢索（cosine，21×1024 numpy 矩陣） |
| bm25 | 手寫 BM25，character bigram 斷詞，Lucene 版 IDF |
| hybrid | 上述兩者以 RRF（Reciprocal Rank Fusion）融合排名，`rrf_k=10` |
| alias | chunk 標頭嵌入該欄位的中英別名（`顯示晶片 (Graphics / GPU / 顯卡 / 獨顯 …)`），供混合語言提問命中 |
| SKU routing | AM6H 有 BZH / BYH / BXH 三種機型，僅顯示晶片有差異。提問指定機型時只取該機型的 chunk，未指定時改取三機型並列的比較 chunk |
| overview chunk | 一筆由各欄位首行組成的規格總覽，用於回應籠統提問 |

語料共 21 筆 chunk：16 筆三機型共用、3 筆顯示晶片各機型專屬、1 筆機型比較、1 筆規格總覽。

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

整體來看，加入別名的 hybrid retrieval 表現最好，23 題可回答問題所需的資料都能出現在前三名。別名是影響最大的設計，移除後 R@3 從 1.000 降至 0.768，表示使用者的口語問法需要額外的同義詞才能穩定對應到正式規格欄位。單獨使用 dense retrieval 或 BM25 都仍有漏查，但兩者合併後可以互補；dense retrieval 負責語意相近的問題，BM25 則更容易命中明確關鍵字。分題型來看，跨欄位與型號比較已達到理論上限，剩餘錯誤主要來自語意相近的單欄位規格。至於規格總覽 chunk，移除後結果沒有下降，代表目前評測題目尚未涵蓋需要整體摘要的問題。

### 生成結果

Tesla T4，`temperature=0`，90 次生成（30 題 × 3 設定）。
`eval/results/gen_eval.json` 保存的是本機 macOS 執行的逐題明細，其 `environment` 欄位標示執行環境；T4 的完整輸出見 notebook。

```
config      facts   part  refuse falseR  fab   簡  TTFT50  TTFT95    TPS
------------------------------------------------------------------------
no RAG      0.174  0.337   0.143  0.043    1    3   0.06s   0.08s   56.8
RAG k=1     0.696  0.819   0.857  0.435    0    1   0.08s   0.17s   52.7
RAG k=3     1.000  1.000   1.000  0.000    0    0   0.34s   0.49s   52.7
```

**分析：**

整體結果顯示，RAG 在取回三筆資料（k=3）時表現最穩定，所有可回答問題都能根據規格頁正確作答，7 題資料中沒有答案的問題也都有適當拒答，沒有出現誤拒、捏造或回答過度簡略的情況。相較之下，未使用 RAG 時，模型經常會自行補上看似合理但實際錯誤的資訊，例如將電池容量回答為 92.5Wh、記憶體上限回答為 128GB，甚至在頁面沒有資料的情況下直接給出售價和保固年限。當檢索數量縮減為 k=1 後，雖然模型仍沒有捏造內容，但因為第一筆結果不一定包含真正需要的欄位，誤拒率上升到 0.435。這表示問題主要不在生成模型，而在於提供的檢索內容是否完整；對這份規格資料而言，k=3 能在資訊覆蓋與輸入長度之間取得較合適的平衡。


### 延遲分析

| | TTFT p50 | TTFT p95 | TPS |
|---|---|---|---|
| 無 RAG（prompt 中位數 78 token） | 0.06s | 0.08s | 56.8 |
| RAG k=1（prompt 中位數 517 token） | 0.08s | 0.17s | 52.7 |
| RAG k=3（prompt 中位數 906 token） | 0.34s | 0.49s | 52.7 |

延遲結果顯示，首字出現的時間主要受到 prompt 長度影響。當 prompt 從無 RAG 的 78 tokens 增加到 RAG k=3 的 906 tokens 時，TTFT p50 從 0.06 秒上升到 0.34 秒，但生成速度只從每秒 56.8 tokens 降至 52.7 tokens，差距不大。換句話說，增加 context 主要拖慢的是模型開始回答前的處理時間，而不是後續生成文字的速度。若要進一步降低延遲，比起更換模型，更直接的做法是精簡 chunk 內容和標頭，或減少取回的資料數量。不過本次實驗中，k=3 雖然比 k=1 多出約 0.26 秒的首字等待時間，卻能將誤拒率從 0.435 降到 0；對規格查詢而言，這樣的延遲增加仍在可接受範圍內，也比因資料不足而無法回答更值得取捨。

### 量測方法的限制

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
