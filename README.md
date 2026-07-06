# RAG 文件知识库

一个本地轻量级知识库系统，支持：

- TXT / PDF / CSV 上传解析
- OpenAI / Claude / DeepSeek / Qwen / 自定义 OpenAI 兼容接口
- 多查询召回：原问题、本地通用查询扩展、可选 LLM 查询改写共同参与检索
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

3. 在主页面打开 `系统设置 ` 面板

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

当前系统检索层采用：

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
  导入路径的适配层
- `data/chroma`
  Chroma 本地持久化目录
- `data/uploads`
  上传文件缓存目录

## PyCharm / Windows 平台

- PyCharm 中请运行：`streamlit run app.py`
- 如果机器不能联网，可提前下载模型并在 UI 中把 `向量模型名或本地模型路径` 改成本地目录；仅临时离线演示时才建议手动使用 `local-hashing-1024`
- 扫描版或图片型 PDF 已内置 OCR ，不需要额外安装 Tesseract；如 OCR 报依赖缺失，请重新运行 `pip install -r requirements.txt`


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
