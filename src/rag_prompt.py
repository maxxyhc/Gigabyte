"""Prompt construction for the answering step.

Kept apart from rag.py so the wording can be revised — and ablated — without
touching the pipeline that calls it.

Answers carry no inline source labels. Attribution lives in the retrieved chunk
ids, which rag.py prints with --sources; putting it in the prose made a 4B model
consolidate the labels into one trailing tag and pick the wrong field for it.

The context is rebuilt from the chunk's fields rather than reusing chunk['text']
verbatim. That text carries the alias header, up to eleven synonyms per field,
which exists to be matched by the retrievers and would only spend context
tokens and invite the model to echo them back.
"""

from __future__ import annotations

SYSTEM = """你是 GIGABYTE AORUS MASTER 16 AM6H 的規格查詢助理。

作答規則：
1. 語言先決定：問題以英文提出時整段回答使用英文，資料未涵蓋時也用英文說明，例如 "The specification page does not provide this information."；其餘一律使用臺灣繁體中文用字（例如「記憶體」而非「内存」、「顯示記憶體」而非「显存」）。
2. 只使用「參考資料」裡的內容。資料中沒有的項目，說明規格頁未提供該項資訊，並停在那裡。
3. 寫成一段連貫的文字，把相關規格組織在同一段裡，用完整的句子敘述。
4. 這台筆電有 BZH / BYH / BXH 三種機型。資料標示「適用機型」時，說清楚該規格屬於哪些機型。
5. 直接作答，不重述問題。

作答範例：

Q：這台的螢幕和網路規格？
A：這台搭載 16 吋 16:10 的 OLED WQXGA 面板，解析度 2560×1600、更新率 240Hz、反應時間 1ms，並支援 NVIDIA G-SYNC 與 Dolby Vision。無線網路方面支援 WIFI 7 (802.11be 2x2)，另有 1G 有線網路與 Bluetooth v5.4。

Q: What graphics card does it use?
A: The graphics card depends on the model. The BZH ships with an NVIDIA GeForce RTX 5090 Laptop GPU and 24GB of GDDR7, while the BXH uses an RTX 5070 Ti Laptop GPU with 12GB of GDDR7."""


# The no-RAG baseline for the evaluation. It withholds the spec context and
# the grounding rules, so what it measures is what the bare 4B model believes
# about a laptop announced after its training data — the reason RAG is here.
PLAIN_SYSTEM = """你是筆電產品助理，回答使用者關於 GIGABYTE AORUS MASTER 16 AM6H 的問題。
問題以英文提出時整段回答使用英文，其餘一律使用臺灣繁體中文用字。"""


def build_plain_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": PLAIN_SYSTEM},
        {"role": "user", "content": question},
    ]


def format_context(hits) -> str:
    blocks = []
    for number, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        scope = "全機型" if chunk["shared_across_models"] else " / ".join(chunk["model_codes"])
        blocks.append(
            f"【資料 {number}】{chunk['field']} ({chunk['field_en']})｜適用機型：{scope}\n"
            f"{chunk['value']}"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, hits) -> list[dict]:
    if not hits:
        # An empty context still goes to the model rather than short-circuiting
        # to a canned line: rule 1 already covers it, and one refusal path is
        # easier to evaluate than two.
        context = "（沒有找到相關資料）"
    else:
        context = format_context(hits)

    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"參考資料：\n\n{context}\n\n問題：{question}"},
    ]
