---
name: smart-recipe-system
description: "构建并运行智能食谱推荐系统。当用户需要根据食材或菜名生成详细食谱、将食谱信息语音播报时使用此技能。基于 Qwen3-VL (多模态食谱生成) + Qwen3-TTS (语音合成) 的 OpenVINO 本地推理。"
---

# 智能食谱推荐系统

基于 OpenVINO 本地推理，通过 Qwen3-VL（食谱生成与信息提取）→ Qwen3-TTS（语音合成播报）管线，帮助用户根据食材或菜名快速生成美味食谱，并可语音播报。

## 系统架构

```
用户输入（食材或菜名）
      │
      ▼
[1] 加载输入
      │
      ▼
[2] VLM 食谱生成 (Qwen3-VL) ──→ 结构化食谱信息
      │  （释放 VLM 模型）
      ▼
[3] TTS 语音合成 (Qwen3-TTS) ──→ 语音音频
      │  （释放 TTS 模型）
      ▼
输出: {recipe_text, audio}
```

两个模型（VLM + TTS）同时加载需大量内存。`ModelManager` 采用**懒加载**策略：每个模型按需加载、用完即释放，以时间换空间。

## 项目结构

项目代码位于 `OpenVINO-smart_recipe_system/` 目录：

```
OpenVINO-smart_recipe_system/
├── gradio_helper.py              # 管线主逻辑 & Gradio 界面（含 ModelManager、smart_recipe_pipeline、make_demo）
├── smart_recipe_system.py        # 智能食谱系统核心逻辑
├── qwen_3_tts_helper.py          # Qwen3-TTS OpenVINO 推理封装
├── notebook_utils.py             # Jupyter 工具函数（设备选择 widget 等）
├── smart_recipe_system.ipynb     # 主入口 Notebook
├── requirements.txt              # Python 依赖
└── README.md                     # 项目说明
```

## 核心模型

| 模型 | 用途 | ModelScope ID | 精度 |
|------|------|---------------|------|
| Qwen3-VL-4B-Instruct | 多模态食谱生成 | `Qwen3-VL-4B-Instruct-int4-ov` | INT4 |
| Qwen3-TTS-CustomVoice-0.6B | 语音合成 | `Qwen3-TTS-CustomVoice-0.6B-fp16-ov` | FP16 |

所有模型均为预转换的 OpenVINO IR 格式（`.xml` + `.bin`），从 ModelScope 下载后直接加载推理。

## 第一步：下载代码并安装依赖

先从 GitHub 克隆项目代码，再安装依赖：

```bash
git clone  https://github.com/z12a/OpenVINO-smart_recipe_system.git
cd OpenVINO-smart_recipe_system
pip install -r requirements.txt
```

如果目录已存在则跳过克隆：

```python
import os
if not os.path.exists('gradio_helper.py') or not os.path.exists('requirements.txt'):
    os.system('git clone  https://github.com/z12a/OpenVINO-smart_recipe_system.git')
    os.chdir('OpenVINO-smart_recipe_system')
```

 
```

 

## 第二步：下载模型

两个模型均从 ModelScope 下载，已转换为 OpenVINO 格式。如果模型目录已存在则跳过下载。

```python
from pathlib import Path
from modelscope import snapshot_download

# --- VLM 模型 ---
vlm_model_dir = Path("Qwen3-VL-4B-Instruct-int4-ov")
if not vlm_model_dir.exists():
    snapshot_download("Qwen3-VL-4B-Instruct-int4-ov", local_dir=str(vlm_model_dir))

# --- TTS 模型 ---
tts_model_dir = Path("Qwen3-TTS-CustomVoice-0.6B-fp16-ov")
if not tts_model_dir.exists():
    snapshot_download("Qwen3-TTS-CustomVoice-0.6B-fp16-ov", local_dir=str(tts_model_dir))
```

## 第三步：运行管线

### 方式 A：Python 脚本（推荐）

使用 `ModelManager` 进行懒加载，模型按需加载、用完即释放：

```python
from gradio_helper import ModelManager, smart_recipe_pipeline

mgr = ModelManager(
    vlm_model_dir="Qwen3-VL-4B-Instruct-int4-ov",
    tts_model_dir="Qwen3-TTS-CustomVoice-0.6B-fp16-ov",
    device="AUTO",
)

result = smart_recipe_pipeline(mgr, user_input="我想吃番茄和鸡蛋")
print(result["recipe_text"])
# result["audio"] -> (sample_rate, wav_data) 或 None
```

### 方式 B：分步调用

```python
from gradio_helper import ModelManager, vlm_generate_recipe, tts_synthesize

mgr = ModelManager(
    vlm_model_dir="Qwen3-VL-4B-Instruct-int4-ov",
    tts_model_dir="Qwen3-TTS-CustomVoice-0.6B-fp16-ov",
    device="AUTO",
)

# Step 1: 加载输入
user_input = "推荐一个川菜"

# Step 2: VLM 食谱生成
vlm_model, vlm_processor = mgr.get_vlm_model()
recipe_text = vlm_generate_recipe(vlm_model, vlm_processor, user_input)
mgr.release_vlm()  # 释放 VLM 模型内存

# Step 3: 语音合成
tts_model = mgr.get_tts_model()
wav_data, sr = tts_synthesize(tts_model, recipe_text, speaker="vivian", language="Chinese")
mgr.release_tts()  # 释放 TTS 模型内存
```

### 方式 C：Gradio Web 界面

```python
from gradio_helper import ModelManager, make_demo

mgr = ModelManager(
    vlm_model_dir="Qwen3-VL-4B-Instruct-int4-ov",
    tts_model_dir="Qwen3-TTS-CustomVoice-0.6B-fp16-ov",
    device="AUTO",
)

demo = make_demo(mgr)
demo.launch(server_port=7860)
```

### 方式 D：Jupyter Notebook

打开 `smart_recipe_system.ipynb`，按顺序执行所有单元格即可。

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `device` | `"AUTO"` | 推理设备（AUTO/CPU/GPU） |
| `vlm_max_new_tokens` | `1024` | VLM 最大生成 token 数 |
| `tts_max_new_tokens` | `2048` | TTS 最大生成 token 数 |
| `tts_speaker` | `"vivian"` | TTS 说话人（vivian/aiden/ryan/serena 等） |
| `tts_language` | `"Chinese"` | TTS 语言（Chinese/English/Japanese 等） |
| `tts_instruct` | `"用友好亲切的语气说话。"` | TTS 风格指令 |
| `release_between_steps` | `True` | 步骤间释放模型以节省内存 |

## 生成的食谱内容

大模型会根据用户输入生成并整理以下内容，用简洁通俗的语言表述：

1. 菜名（或推荐的菜名）
2. 菜品简介（菜品特点和风格）
3. 所需食材（包含主要食材、调料和分量）
4. 烹饪步骤（分步骤列出烹饪过程）
5. 小贴士（提供烹饪技巧或替代食材建议）

## 核心 API 参考

### ModelManager

```python
from gradio_helper import ModelManager

mgr = ModelManager(vlm_model_dir, tts_model_dir, device="AUTO")

mgr.get_vlm_model()       # 懒加载 VLM 模型，返回 (model, processor)
mgr.get_tts_model()       # 懒加载 TTS 模型

mgr.release_vlm()         # 释放 VLM 模型内存
mgr.release_tts()         # 释放 TTS 模型内存
mgr.release_all()         # 释放所有模型
```

### smart_recipe_pipeline

```python
from gradio_helper import smart_recipe_pipeline

result = smart_recipe_pipeline(
    model_manager=mgr,
    user_input="我想吃番茄和鸡蛋",
    vlm_max_new_tokens=1024,
    tts_max_new_tokens=2048,
    tts_speaker="vivian",
    tts_language="Chinese",
    tts_instruct="用友好亲切的语气说话。",
    release_between_steps=True,
)
# result["recipe_text"]   -> VLM 生成的食谱文本
# result["audio"]         -> (sample_rate, wav_data) 或 None
```

### 独立函数

```python
from gradio_helper import vlm_generate_recipe, tts_synthesize

vlm_generate_recipe(vlm_model, vlm_processor, user_input, max_new_tokens=1024)  # 食谱生成
tts_synthesize(tts_model, text, speaker="vivian", language="Chinese", instruct="...", max_new_tokens=2048)  # 语音合成
```

## 快速检查清单

1. 依赖已安装：`pip install -r requirements.txt`
2. 系统库已安装（Linux）：`ldconfig -p | grep libGL`
3. 模型目录存在且完整（含 `.xml` + `.bin` 文件）：
   - `Qwen3-VL-4B-Instruct-int4-ov/`
   - `Qwen3-TTS-CustomVoice-0.6B-fp16-ov/`
4. OpenVINO 版本：`python -c "import openvino; print(openvino.__version__)"` 应 >= 2025.4
5. 可用设备：`python -c "import openvino as ov; print(ov.Core().available_devices)"`

## 常见错误排查

| 错误 | 原因 | 解决方法 |
|------|------|----------|
| `ModuleNotFoundError: smart_recipe_system` | 工作目录不对 | `cd` 到 `OpenVINO-smart_recipe_system/` 目录后再运行 |
| `ModuleNotFoundError: gradio_helper` | 工作目录不对 | 确认在 `OpenVINO-smart_recipe_system/` 目录下运行 |
| `ModuleNotFoundError: optimum` | optimum-intel 未安装 | 按 requirements.txt 安装 |
| `FileNotFoundError: model_dir` | 模型未下载 | 运行模型下载代码或手动 `snapshot_download` |
| Gradio 端口占用 | 7860 端口被占用 | `demo.launch(server_port=7861)` |
| 食谱生成失败 | 输入文本过长或含特殊字符 | `gradio_helper.clean_for_vlm()` 会自动清理 |
| TTS 合成失败 | 文本过长或含特殊字符 | `gradio_helper.clean_for_tts()` 会自动清理 |
| `ImportError: libGL.so.1` | 缺少 OpenGL 库 | `sudo apt-get install libgl1-mesa-glx` |
| 内存不足 | 多个模型同时加载 | 确保 `release_between_steps=True` |

## 使用场景示例

### 场景 1：根据食材推荐菜肴

```python
result = smart_recipe_pipeline(mgr, user_input="我有鸡蛋、番茄和洋葱")
print(result["recipe_text"])
```

### 场景 2：推荐特定地域菜系

```python
result = smart_recipe_pipeline(mgr, user_input="推荐一个四川菜，适合夏天吃")
print(result["recipe_text"])
```

### 场景 3：生成食谱并播报

```python
result = smart_recipe_pipeline(
    mgr, 
    user_input="请推荐一个简单快手菜",
    tts_speaker="aiden"  # 不同的说话人
)
# result["audio"] 包含生成的语音
```

### 场景 4：自定义 TTS 风格

```python
result = smart_recipe_pipeline(
    mgr,
    user_input="我想吃番茄鸡蛋面",
    tts_instruct="用活泼热情的语气说话，适合给小朋友讲故事。"
)
```
