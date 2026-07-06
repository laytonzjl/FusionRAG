# RAG 文件知识库

一个本地轻量级企业知识库系统，支持：

- Streamlit 对话式交互
- TXT / PDF / CSV 上传解析
- Chroma 本地持久化向量库
- 聊天模型与本地向量模型分离配置
- OpenAI / Claude / DeepSeek / Qwen / 自定义 OpenAI 兼容接口
- 默认使用真实本地语义向量 `intfloat/multilingual-e5-small`，不需要向量 API Key
- 多查询召回：原问题、本地通用查询扩展、可选 LLM 查询改写共同参与检索
- PDF 原生文本解析 + 扫描页 OCR 兜底识别
- 可热配置的 UI 设置、切片参数、检索参数和向量库参数
- 引用来源卡片、回答置信度与来源标签

## 启动方式

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 运行应用

```bash
streamlit run app.py
```

3. 在主页面打开 `运行配置 v3` 面板

- 推荐点击 `一键应用并保存：DeepSeek 聊天 + 本地向量`
- 填写聊天模型 API Key
- 本地向量不需要 API Key，向量接口地址固定为 `local`
- 默认向量模型为 `intfloat/multilingual-e5-small`，首次使用可能需要下载模型文件
- 按需调整切片大小、切片重叠、Top-K、候选片段数、知识库集合名
- 如新资料领域跨度较大，可打开 `启用 LLM 查询改写`；系统会生成少量检索改写问题再多路召回
- 如果导入扫描版 PDF，请保持 `启用 PDF OCR` 打开；默认会对低文本页自动识别
- OCR 默认使用 CPU。如果扫描 PDF 很慢，可在设置中把 `OCR 推理设备` 改为 `DirectML` 或 `CUDA`，并按下方 GPU 说明安装对应运行时
- 点击 `保存运行配置`

说明：

- 支持热配置，保存后立即生效，不需要手工先改 `.env`
- API 配置会保存到 `data/api_config.json`
- RAG 参数会保存到 `data/rag_config.json`
- UI 偏好会保存到 `data/ui_config.json`
- `.env` 只是可选的初始默认值来源
- 如果刚从 OpenAI 向量切到本地向量，请新建知识库集合名或点击 `清空当前知识库` 后重新导入文件

## 检索策略说明

当前系统不依赖针对某本书、某个人物或某个地名的手写知识规则。检索层采用：

- 查询焦点抽取：从用户问题自动提取核心主题词，用于跨文档降噪
- 通用意图扩展：只扩展年龄、地点、定义、原因、方法等通用问法
- 多查询向量召回：原问题和扩展问题分别召回后融合排序
- BM25 词法召回：补足向量模型对短问题、专有名词和 OCR 噪声的不足
- 证据感知排序：封面、CIP、目录、前言、附录等非正文内容默认降权
- 可选 LLM 查询改写：关闭时零额外模型调用，开启后泛化更强但会略慢

## 目录结构

- `app.py`
  Streamlit 启动入口
- `config.py`
  统一管理 API、向量、RAG、UI 配置及本地持久化
- `rag_ui/`
  Streamlit 界面层
- `rag_core/`
  文档解析、切片、向量化、检索、问答核心逻辑
- `rag_engine.py`
  兼容旧导入路径的适配层
- `data/chroma`
  Chroma 本地持久化目录
- `data/uploads`
  上传文件缓存目录

## PyCharm / Windows 注意

- PyCharm 中请运行：`streamlit run app.py`，不要直接运行 `python app.py`
- 修改 UI 代码后，如果页面没有出现 `运行配置 v3`，请停止 PyCharm 当前运行进程后重新启动
- 如果机器不能联网，可提前下载模型并在 UI 中把 `向量模型名或本地模型路径` 改成本地目录；仅临时离线演示时才建议手动使用 `local-hashing-1024`
- 扫描版或图片型 PDF 已内置 OCR 兜底，不需要额外安装 Tesseract；如 OCR 报依赖缺失，请重新运行 `pip install -r requirements.txt`

## OCR GPU 加速

当前 OCR 基于 RapidOCR + ONNX Runtime。默认 `requirements.txt` 使用 CPU 运行时，最稳但扫描 PDF 会慢。

Windows 推荐优先使用 DirectML，兼容 NVIDIA / AMD / Intel 显卡：

```bash
python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install -r requirements-ocr-gpu-directml.txt
```

NVIDIA CUDA 用户也可以使用 CUDA 运行时，但需要本机 CUDA / cuDNN 与 `onnxruntime-gpu` 版本匹配：

```bash
python -m pip uninstall -y onnxruntime onnxruntime-directml
python -m pip install -r requirements-ocr-gpu-cuda.txt
```

安装后重启 Streamlit，并在右上角 `设置` 中调整：

- `OCR 推理设备`：优先试 `DirectML GPU（Windows 通用）`，NVIDIA CUDA 环境可试 `CUDA GPU（NVIDIA）`
- `OCR 渲染 DPI`：越高越慢，默认 150；扫描文本推荐 150-180，细小表格可提高到 220
- `OCR 最大图像边长`：越小越快，默认 1400；小说/合同推荐 1400-1800，复杂版面可提高
- `OCR CPU 线程数`：CPU 模式可试 4、6、8；GPU 模式保持 `-1` 或较小值即可

如果 UI 中显示可用 Provider 只有 `CPUExecutionProvider`，说明当前环境还没有真正启用 GPU OCR。

## 当前这版重构重点

- UI 改成更接近 OpenAI 的简洁工作台风格
- 默认使用本地向量，避免 DeepSeek / Claude / Qwen 聊天 Key 被误用于 OpenAI 向量接口
- 聊天供应商与向量供应商彻底解耦，并提供可见保存按钮
- 不同供应商切换时，默认接口地址和模型名会自动同步到正确值
- Chroma 集合名、距离度量、切片参数、检索参数都可以在 UI 中配置
- PDF OCR 开关、渲染 DPI、低文本页触发阈值、OCR 推理设备、线程数和最大图像边长都可以在 UI 中配置

## 多语言证据优先混合检索 v4

本版本将检索链路升级为 `Multilingual Document-Adaptive Evidence-First Hybrid RAG`。系统不再把向量相似度直接当成答案依据，而是先规划问题、再多路召回、融合、重排、回填上下文，最后只基于可追溯证据回答。

入库时会额外构建：

- `Document Card`：文档标题、作者/发布方、版本、主题、证据页码等基础信息卡片
- `Entity Card`：人物、机构、地点、术语、作品、章节、代码符号等实体与别名卡片
- `Parent / Child Chunk`：父子层级切块，子块用于召回，父块用于上下文回填
- `SQLite Hybrid Index`：本地词法、精确实体、结构化索引，默认位置为 `data/rag_hybrid.sqlite`
- `Chroma Dense Index`：继续保留现有本地向量库，用于多语言 dense retrieval

在线问答时会执行：

- Query Planner：识别问题语言、意图、实体、别名与检索变体
- Exact Retrieval：优先命中文档标题、章节标题、实体别名、代码符号、条款编号
- Language-Aware Lexical Retrieval：中文/日文/韩文使用字符 n-gram，拉丁语言使用 Unicode token，混合文本保留代码、路径、版本号
- Dense Retrieval：继续使用 Chroma 与本地 embedding；E5 系列自动区分 `query:` 与 `passage:`
- Weighted RRF：按 rank 融合多检索通道，不直接相加不同检索器原始分数
- Cross-Encoder Rerank：可选，默认模型配置为 `BAAI/bge-reranker-v2-m3`
- Evidence Judge：可选高准确模式，只判断证据相关性，不生成最终答案

## 旧知识库迁移

旧 Chroma collection 仍可读取，但旧索引缺少 `Document Card`、`Entity Card`、父子切块、SQLite 词法索引和多语言 metadata，因此短实体问题与跨语言问题会自动降级为向量检索。升级后建议重建一次当前知识库。

方式一：在 UI 中打开右上角知识库面板，点击 `从已保存上传文件重建索引`。

方式二：命令行重建：

```bash
python scripts/rebuild_index.py
```

重建会清空当前 Chroma collection 和 `data/rag_hybrid.sqlite` 中对应 collection 的记录，但不会删除 `data/uploads` 中已经保存的原始上传文件。

如果修改了以下关键参数，系统会在知识库管理面板提示索引不兼容，需要重建：

- embedding provider / embedding model
- parent chunk size / child chunk size / overlap
- normalization version
- language processing version
- lexical index version
- document structure version

## 运行模式建议

快速模式：

- 启用混合检索、精确实体检索、结构化检索
- 关闭 Cross-Encoder Rerank
- 关闭 LLM Evidence Judge
- 适合本地离线演示和大批量 PDF 初筛

准确模式：

- 启用混合检索、精确实体检索、结构化检索、Query Planner
- 启用 Cross-Encoder Rerank
- 关闭或按需启用 LLM Evidence Judge
- 适合人物关系、合同条款、技术文档问答

跨语言模式：

- 启用 Query Planner
- 启用跨语言检索变体
- 使用多语言 embedding，例如 `intfloat/multilingual-e5-small` 或 `BAAI/bge-m3`
- 可启用多语言 reranker，例如 `BAAI/bge-reranker-v2-m3`
- 回答会使用用户问题语言，引用仍保留原始文档语言

## 检索诊断

每次回答后，页面会显示可展开的 `检索诊断`：

- Query Plan
- 问题语言与置信度
- 识别实体、别名与来源
- 检索变体与语言
- Exact / Structured / Lexical / Dense 各通道候选数
- RRF 贡献
- Cross-Encoder 分数
- Evidence Judge 结论
- 最终证据块、页码、语言和 `entity_coverage_failed`

诊断面板不会展示 API Key 或完整系统提示词。
