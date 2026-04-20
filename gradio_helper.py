"""
Gradio helper for Smart Recipe Recommendation System.
Provides a ModelManager for lazy model loading, and a make_demo function
to create the Gradio interface for ingredient recognition, recipe generation,
and TTS playback.
"""

import gc
import logging
import re
import tempfile
import time
import math

import numpy as np
import gradio as gr
from PIL import Image
from scipy.io.wavfile import write

logger = logging.getLogger("recipe_recommender")

# Add support for qwen3_vl model type
try:
    from optimum.intel.openvino.modeling_visual_language import MODEL_TYPE_TO_CLS_MAPPING, _OVQwen2VLForCausalLM
    MODEL_TYPE_TO_CLS_MAPPING['qwen3_vl'] = _OVQwen2VLForCausalLM
except ImportError:
    pass

# Default prompts and generation config used by the recipe recommender when
# the notebook does not supply its own values.
DEFAULT_SYSTEM_PROMPT = """你是一位世界顶级的中餐大厨，拥有丰富的烹饪经验和营养学知识。请根据用户提供的食材信息，生成一份详细的个性化食谱。要求：菜品名称要吸引人且准确描述菜品特色；食材清单要包含具体分量；烹饪步骤要清晰详细，适合家庭厨房操作；提供营养分析和健康建议；给出烹饪小贴士和替代方案。"""

DEFAULT_GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": False,
    "repetition_penalty": 1.0,
}


def clean_for_tts(text):
    """
    Clean text for TTS synthesis by removing content that cannot be
    properly synthesized into speech, such as emojis and markdown formatting.

    Args:
        text: Input text string

    Returns:
        str: Cleaned text suitable for TTS
    """
    # Remove emojis (Unicode ranges for common emojis)
    # NOTE: Must avoid ranges that overlap with CJK characters (U+4E00-U+9FFF)
    text = re.sub(
        r"[\U0001F600-\U0001F64F"  # emoticons
        r"\U0001F300-\U0001F5FF"   # symbols & pictographs
        r"\U0001F680-\U0001F6FF"   # transport & map
        r"\U0001F1E0-\U0001F1FF"   # flags
        r"\U00002702-\U000027B0"   # dingbats
        r"\U000024C2-\U0000324F"   # enclosed alphanumerics (stop before CJK)
        r"\U0001F200-\U0001F251"   # enclosed CJK supplement (above CJK range)
        r"\U0001F900-\U0001F9FF"   # supplemental symbols
        r"\U0001FA00-\U0001FA6F"   # chess symbols
        r"\U0001FA70-\U0001FAFF"   # symbols extended-A
        r"\U00002600-\U000026FF"   # misc symbols
        r"\U0000FE00-\U0000FE0F"   # variation selectors
        r"\U0000200D"              # zero-width joiner
        r"]+",
        "",
        text,
    )
    # Remove markdown code blocks (```...```)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # Remove inline code (`...`) -> content
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    # Remove markdown headers (# ## ### etc.) at line start
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove markdown bold (**text**) -> text
    text = re.sub(r"\*\*([^*\n]+?)\*\*", r"\1", text)
    # Remove markdown bold (__text__) -> text
    text = re.sub(r"__([^_\n]+?)__", r"\1", text)
    # Remove markdown italic (*text*) -> text
    text = re.sub(r"\*([^*\n]+?)\*", r"\1", text)
    # Remove markdown italic (_text_) -> text (only when _ is at word boundary)
    text = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"\1", text)
    # Remove markdown links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove markdown images ![alt](url)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    # Remove markdown horizontal rules (---, ***, ___)
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Remove markdown bullet list markers (- , * , + ) at line start, keep content
    text = re.sub(r"^(\s*)[-*+]\s+", r"\1", text, flags=re.MULTILINE)
    # Remove markdown numbered list markers (1. 2. etc.) at line start, keep content
    text = re.sub(r"^(\s*)\d+\.\s+", r"\1", text, flags=re.MULTILINE)
    # Remove markdown table pipes
    text = re.sub(r"\|", " ", text)
    # Remove markdown table separator lines (---:---:---)
    text = re.sub(r"^[-: ]+$", "", text, flags=re.MULTILINE)
    # Collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    # Remove leading/trailing whitespace overall
    text = text.strip()
    return text


class ModelManager:
    """
    Lazy model manager that loads models on demand and releases them after use.

    Instead of loading all 3 models (OCR + VLM + TTS) into memory at once,
    this manager loads each model only when needed and optionally releases
    the previous model before loading the next one to save memory.

    Usage:
        mgr = ModelManager(
            ocr_model_dir="PaddleOCR-VL-1.5-OpenVINO",
            vlm_model_dir="Qwen3-VL-4B-Instruct-int4-ov",
            tts_model_dir="Qwen3-TTS-CustomVoice-0.6B-fp16-ov",
            device="AUTO",
        )

        ocr_model = mgr.get_ocr_model()   # OCR model loaded now
        vlm_model, vlm_proc = mgr.get_vlm_model()  # VLM model loaded now
        tts_model = mgr.get_tts_model()   # TTS model loaded now

        mgr.release_ocr()   # Free OCR model memory
        mgr.release_vlm()   # Free VLM model memory
        mgr.release_tts()   # Free TTS model memory
    """

    def __init__(
        self,
        ocr_model_dir,
        vlm_model_dir,
        tts_model_dir,
        device="AUTO",
    ):
        self.ocr_model_dir = str(ocr_model_dir)
        self.vlm_model_dir = str(vlm_model_dir)
        self.tts_model_dir = str(tts_model_dir)
        self.device = self._normalize_device(device)

        # Lazy-loaded model instances
        self._ocr_model = None
        self._vlm_model = None
        self._vlm_processor = None
        self._tts_model = None

        # Shared OpenVINO core
        self._ov_core = None

    def _normalize_device(self, device):
        """Normalize device input into a string for OpenVINO."""
        if device is None:
            return "AUTO"
        if hasattr(device, "value"):
            device = device.value
        if isinstance(device, str):
            return device
        try:
            return str(device)
        except Exception:
            return "AUTO"

    def _get_ov_core(self):
        if self._ov_core is None:
            import openvino as ov
            self._ov_core = ov.Core()
        return self._ov_core

    def get_ocr_model(self):
        """Get PaddleOCR-VL model, loading it lazily if not already loaded."""
        if self._ocr_model is None:
            logger.info("加载 OCR 模型 (PaddleOCR-VL)...")
            start = time.perf_counter()

            import openvino as ov
            from ov_paddleocr_vl import OVPaddleOCRVLForCausalLM

            core = self._get_ov_core()
            self._ocr_model = OVPaddleOCRVLForCausalLM(
                core=core,
                ov_model_path=self.ocr_model_dir,
                device=self._normalize_device(self.device),
                llm_int4_compress=False,
                llm_int8_compress=True,
                vision_int8_quant=False,
                llm_int8_quant=True,
                llm_infer_list=[],
                vision_infer=[],
            )

            elapsed = time.perf_counter() - start
            logger.info("OCR 模型加载完成, 耗时: %.2fs", elapsed)
        return self._ocr_model

    def get_vlm_model(self):
        """Get VLM model and processor, loading lazily if not already loaded."""
        if self._vlm_model is None or self._vlm_processor is None:
            logger.info("加载 VLM 模型 (Qwen3-VL)...")
            start = time.perf_counter()

            from optimum.intel.openvino import OVModelForVisualCausalLM
            from transformers import AutoConfig, AutoProcessor

            device = self._normalize_device(self.device)
            config = AutoConfig.from_pretrained(self.vlm_model_dir, trust_remote_code=True)
            # Handle unsupported model types
            if config.model_type == 'qwen3_vl':
                config.model_type = 'qwen2_vl'
            if not hasattr(config.vision_config, "embed_dim"):
                if hasattr(config.vision_config, "hidden_size"):
                    config.vision_config.embed_dim = config.vision_config.hidden_size
                elif hasattr(config.vision_config, "out_hidden_size"):
                    config.vision_config.embed_dim = config.vision_config.out_hidden_size
                else:
                    config.vision_config.embed_dim = 1024
            self._vlm_model = OVModelForVisualCausalLM.from_pretrained(
                self.vlm_model_dir,
                config=config,
                device=device,
                trust_remote_code=True,
            )
            self._vlm_processor = AutoProcessor.from_pretrained(
                self.vlm_model_dir,
                min_pixels=256 * 28 * 28,
                max_pixels=1280 * 28 * 28,
                fix_mistral_regex=True,
            )

            elapsed = time.perf_counter() - start
            logger.info("VLM 模型加载完成, 耗时: %.2fs", elapsed)
        return self._vlm_model, self._vlm_processor

    def get_tts_model(self):
        """Get TTS model, loading lazily if not already loaded."""
        if self._tts_model is None:
            logger.info("加载 TTS 模型 (Qwen3-TTS)...")
            start = time.perf_counter()

            from qwen_3_tts_helper import OVQwen3TTSModel

            self._tts_model = OVQwen3TTSModel.from_pretrained(
                model_dir=self.tts_model_dir,
                device=self._normalize_device(self.device),
            )

            elapsed = time.perf_counter() - start
            logger.info("TTS 模型加载完成, 耗时: %.2fs", elapsed)
        return self._tts_model

    def release_ocr(self):
        """Release OCR model to free memory."""
        if self._ocr_model is not None:
            logger.info("释放 OCR 模型内存...")
            del self._ocr_model
            self._ocr_model = None
            gc.collect()

    def release_vlm(self):
        """Release VLM model and processor to free memory."""
        if self._vlm_model is not None:
            logger.info("释放 VLM 模型内存...")
            del self._vlm_model
            del self._vlm_processor
            self._vlm_model = None
            self._vlm_processor = None
            gc.collect()

    def release_tts(self):
        """Release TTS model to free memory."""
        if self._tts_model is not None:
            logger.info("释放 TTS 模型内存...")
            del self._tts_model
            self._tts_model = None
            gc.collect()

    def release_all(self):
        """Release all models to free memory."""
        self.release_ocr()
        self.release_vlm()
        self.release_tts()

    def get_tts_speakers_and_languages(self):
        """Get supported TTS speakers and languages.

        Avoid loading the TTS model during Gradio interface construction.
        The UI can still use default speaker/language values and load TTS on demand.
        """
        if self._tts_model is None:
            return ["vivian"], ["Chinese"]

        try:
            return self._tts_model.get_supported_speakers(), self._tts_model.get_supported_languages()
        except Exception:
            return ["vivian"], ["Chinese"]


def split_image(image, num_splits=4, overlap_ratio=0.1):
    """
    Split an image into num_splits parts (NxN grid) with overlap.

    Args:
        image: PIL.Image object
        num_splits: Number of splits, must be a perfect square (e.g. 4=2x2, 9=3x3)
        overlap_ratio: Overlap ratio (0~1) between split regions

    Returns:
        List[PIL.Image]: List of split sub-images
    """
    grid_size = int(math.sqrt(num_splits))
    if grid_size * grid_size != num_splits:
        raise ValueError(f"num_splits must be a perfect square (e.g. 4, 9, 16), got: {num_splits}")

    w, h = image.size
    cell_w = w / grid_size
    cell_h = h / grid_size
    overlap_w = cell_w * overlap_ratio
    overlap_h = cell_h * overlap_ratio

    sub_images = []
    for row in range(grid_size):
        for col in range(grid_size):
            left = max(0, col * cell_w - overlap_w)
            upper = max(0, row * cell_h - overlap_h)
            right = min(w, (col + 1) * cell_w + overlap_w)
            lower = min(h, (row + 1) * cell_h + overlap_h)
            sub_img = image.crop((int(left), int(upper), int(right), int(lower)))
            sub_images.append(sub_img)

    return sub_images


def ocr_recognize(paddleocr_model, image, max_new_tokens=5120):
    """
    Use PaddleOCR-VL model to perform OCR on an image.

    Args:
        paddleocr_model: Loaded PaddleOCR-VL model
        image: PIL.Image object
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        str: OCR recognized text
    """
    logger.info("OCR 识别开始, 图片尺寸: %s, max_new_tokens: %d", image.size, max_new_tokens)
    start = time.perf_counter()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "OCR:"},
            ],
        }
    ]

    generation_config = {
        "bos_token_id": paddleocr_model.tokenizer.bos_token_id,
        "eos_token_id": paddleocr_model.tokenizer.eos_token_id,
        "pad_token_id": paddleocr_model.tokenizer.pad_token_id,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }

    response, _ = paddleocr_model.chat(messages=messages, generation_config=generation_config)

    elapsed = time.perf_counter() - start
    logger.info("OCR 识别完成, 耗时: %.2fs, 识别文字长度: %d", elapsed, len(response))
    return response


def vlm_extract_ingredients(vlm_model, vlm_processor, image=None, ocr_text=None, max_new_tokens=512):
    """Use VLM to identify food items / ingredients from an image (and optional OCR text).

    Returns a concise ingredients list and item descriptions suitable for recipe generation.
    """
    logger.info("VLM 识别食材开始, OCR 长度: %s", len(ocr_text) if ocr_text else 0)
    start = time.perf_counter()

    # For supermarket shelf / packaging images we want more structured output
    # including brand, category, and production/expiry dates when visible.
    prompt_parts = []
    if ocr_text:
        prompt_parts.append(f"参考OCR识别结果：\n{ocr_text}\n")

    # Ask the VLM to output a JSON list of detected items with fields to
    # facilitate downstream recipe recommendation and near-expiry reminders.
    prompt_parts.append(
        "请从图片中识别可见的食材或商品，并以 JSON 数组形式输出每一项，数组中每个对象包含字段：\n"
        "- name: 食材或商品名称（必须）\n"
        "- quantity: 可见数量或包装信息（如果可见，否则为 null）\n"
        "- brand: 品牌（如果可见，否则为 null）\n"
        "- category: 建议的类别（如 蔬菜/肉类/海鲜/乳制品/调味料/零食/饮料 等）\n"
        "- production_date: 生产日期文本（如可见，否则为 null）\n"
        "- expiry_date: 有效期/到期日文本（如可见，否则为 null）\n"
        "- expiry_status: 当能解析出日期时，判断状态（""正常""/""临期""/""已过期""），临期定义为在7天内到期；如果无法判断则为 null。\n"
        "要求：输出严格的 JSON（不要其他多余说明），日期请尽量以 YYYY-MM-DD 或自然语义描述，若无法识别某字段填 null。最后在 JSON 之外输出一行简短（不超过20字）的总体建议，例如：'建议优先使用牛奶和鸡蛋'。"
    )

    prompt = "\n".join(prompt_parts)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image} if image is not None else {"type": "text", "text": ""},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = vlm_processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    gen = vlm_model.generate(**inputs, max_new_tokens=max_new_tokens)

    # decode only new tokens
    input_len = inputs["input_ids"].shape[1]
    new_ids = gen[:, input_len:]
    result = vlm_processor.batch_decode(new_ids, skip_special_tokens=True)[0]
    result = result.strip()

    # 如果返回中包含 JSON，则尽量解析并格式化，同时生成一个供生成菜谱使用的简短食材文本列表
    ingredients_text_for_recipe = None
    try:
        import json

        s = result.find("[")
        e = result.rfind("]") + 1
        if s != -1 and e > s:
            json_text = result[s:e]
            parsed = json.loads(json_text)
            pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
            # 构建简短的食材列表文本供后续菜谱生成使用
            names = []
            for item in parsed:
                n = item.get("name") or item.get("名称") or None
                status = item.get("expiry_status") or item.get("expiry") or None
                if n:
                    if status and status != "正常":
                        names.append(f"{n} (状态:{status})")
                    else:
                        names.append(n)
            ingredients_text_for_recipe = ", ".join(names)
            # replace JSON region with pretty-printed JSON + trailing suggestion (if any)
            tail = result[e:].strip()
            result = pretty + ("\n" + tail if tail else "")
    except Exception:
        # leave result as-is if parsing fails
        ingredients_text_for_recipe = None

    # Also prepare a tts-friendly version
    tts_ready = clean_for_tts(result)

    elapsed = time.perf_counter() - start
    logger.info("VLM 识别食材完成, 耗时: %.2fs, 结果长度: %d", elapsed, len(result))
    # 返回 tuple: 原始结果文本, 供生成菜谱的简短食材文本（可能为 None），和 tts-ready 文本
    return result, ingredients_text_for_recipe, tts_ready


def vlm_generate_recipe(vlm_model, vlm_processor, ingredients_text, generation_config=None):
    """Use the VLM/LLM model to generate a structured recipe from ingredients text.

    Returns a markdown/plain text recipe following OUTPUT_FORMAT where possible.
    """
    logger.info("生成食谱开始, 输入长度: %d", len(ingredients_text))
    start = time.perf_counter()

    gen_cfg = generation_config or DEFAULT_GENERATION_CONFIG

    prompt = (
        DEFAULT_SYSTEM_PROMPT
        + "\n\n根据以下食材信息，生成一份家庭可操作的菜谱，包含菜名、简介、食材与用量、详细步骤、营养分析与小贴士。请按条目清晰输出。\n\n"
        + "食材信息：\n"
        + ingredients_text
    )

    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]

    inputs = vlm_processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    gen = vlm_model.generate(**inputs, max_new_tokens=gen_cfg.get("max_new_tokens", 512))

    input_len = inputs["input_ids"].shape[1]
    new_ids = gen[:, input_len:]
    recipe = vlm_processor.batch_decode(new_ids, skip_special_tokens=True)[0]
    recipe = clean_for_tts(recipe)

    elapsed = time.perf_counter() - start
    logger.info("生成食谱完成, 耗时: %.2fs, 结果长度: %d", elapsed, len(recipe))
    return recipe


def tts_synthesize(tts_model, text, speaker="vivian", language="Chinese", instruct="用友好亲切的语气说话。", max_new_tokens=2048):
    """
    Use Qwen3-TTS model to synthesize speech from text.

    Args:
        tts_model: Loaded TTS model
        text: Text to synthesize
        speaker: Speaker name
        language: Language
        instruct: Style instruction
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        tuple: (wav_data, sample_rate) audio data and sample rate
    """
    logger.info("TTS 语音合成开始, 说话人: %s, 语言: %s, 输入文字长度: %d", speaker, language, len(text))
    start = time.perf_counter()

    wavs, sr = tts_model.generate_custom_voice(
        text=text,
        speaker=speaker,
        language=language,
        instruct=instruct,
        non_streaming_mode=True,
        max_new_tokens=max_new_tokens,
    )

    elapsed = time.perf_counter() - start

    if wavs is not None:
        audio_duration = len(wavs[0]) / sr
        logger.info("TTS 语音合成完成, 耗时: %.2fs, 音频时长: %.2fs, 采样率: %d Hz", elapsed, audio_duration, sr)
        return wavs[0], sr

    logger.warning("TTS 语音合成失败, 耗时: %.2fs", elapsed)
    return None, None


def recipe_recommender_pipeline(
    model_manager,
    image_path=None,
    input_text=None,
    input_type="text",
    enable_split=False,
    num_splits=4,
    overlap_ratio=0.1,
    vlm_max_new_tokens=512,
    tts_max_new_tokens=2048,
    tts_speaker="vivian",
    tts_language="Chinese",
    tts_instruct="用友好亲切的语气说话。",
    release_between_steps=True,
):
    """Smart recipe recommendation pipeline.

    Supports either text input (list of ingredients) or image input (photo of ingredients or shelf).
    Uses VLM to extract ingredients from image (optionally using OCR), then generates a recipe and optional TTS.
    """
    pipeline_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("智能食谱推荐管线启动")
    logger.info("  输入类型: %s", input_type)
    logger.info("  图片路径: %s", image_path)
    logger.info("  步骤间释放模型: %s", release_between_steps)
    logger.info("=" * 60)

    result = {}

    image = None
    combined_ocr_text = None

    # If image input, load image
    if input_type == "image":
        logger.info("[Step 1] 加载图片 %s", image_path)
        image = Image.open(image_path).convert("RGB")

        # optional split (typically not needed for simple ingredient photos)
        if enable_split:
            sub_images = split_image(image, num_splits=num_splits, overlap_ratio=overlap_ratio)
            image_for_vlm = image
        else:
            image_for_vlm = image

    # Step: VLM identification (image) or use provided text
    vlm_model, vlm_processor = model_manager.get_vlm_model()

    if input_type == "image":
        logger.info("[Step 2] 使用 VLM 识别图片中的食材...")
        # try to run OCR if OCR model exists to provide extra context
        try:
            ocr_model = model_manager.get_ocr_model()
            combined_ocr_text = ocr_recognize(ocr_model, image_for_vlm, max_new_tokens=1024)
        except Exception:
            combined_ocr_text = None

        extracted_result, short_ingredients, tts_ready = vlm_extract_ingredients(vlm_model, vlm_processor, image=image_for_vlm, ocr_text=combined_ocr_text, max_new_tokens=vlm_max_new_tokens)
        # ingredients_text will be short_ingredients if available, else fallback to extracted_result text
        ingredients_text = short_ingredients or extracted_result
        # attach extracted json/text for output
        result["extracted_raw"] = extracted_result
        result["extracted_tts"] = tts_ready
    else:
        logger.info("[Step 2] 使用文本输入作为食材信息")
        ingredients_text = input_text or ""

    result["ingredients"] = ingredients_text

    # ingredients already set above for image/text

    if release_between_steps:
        model_manager.release_vlm()
        logger.info("VLM 模型已释放")

    # Step: Generate recipe
    vlm_model, vlm_processor = model_manager.get_vlm_model()
    recipe = vlm_generate_recipe(vlm_model, vlm_processor, ingredients_text, generation_config=None)
    result["recipe"] = recipe

    if release_between_steps:
        model_manager.release_vlm()

    # Optional TTS
    tts_audio = None
    if tts_speaker and tts_language is not None:
        try:
            tts_model = model_manager.get_tts_model()
            wav_data, sr = tts_synthesize(tts_model, recipe, speaker=tts_speaker, language=tts_language, instruct=tts_instruct, max_new_tokens=tts_max_new_tokens)
            if wav_data is not None:
                tts_audio = (sr, wav_data)
        except Exception:
            tts_audio = None

    result["audio"] = tts_audio

    if release_between_steps:
        model_manager.release_tts()

    pipeline_elapsed = time.perf_counter() - pipeline_start
    logger.info("管线执行完成, 总耗时: %.2fs", pipeline_elapsed)

    return result


def make_demo(model_manager, vlm_max_new_tokens=512, tts_max_new_tokens=2048):
    """Create Gradio demo for Smart Recipe Recommender."""

    def gradio_pipeline(
        input_type,
        text_input,
        image_input,
        enable_tts,
        tts_speaker,
        tts_language,
        tts_instruct,
    ):
        """Gradio interface main processing function for recipe recommender"""
        tmp_path = None
        try:
            if input_type == "图片上传":
                if image_input is None:
                    return "请上传食材图片或选择文本输入", "", None

                if isinstance(image_input, str):
                    image = Image.open(image_input).convert("RGB")
                else:
                    image = Image.fromarray(image_input).convert("RGB") if not isinstance(image_input, Image.Image) else image_input

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    image.save(tmp.name)
                    tmp_path = tmp.name

                result = recipe_recommender_pipeline(
                    model_manager=model_manager,
                    image_path=tmp_path,
                    input_type="image",
                    vlm_max_new_tokens=vlm_max_new_tokens,
                    tts_max_new_tokens=tts_max_new_tokens,
                    tts_speaker=tts_speaker,
                    tts_language=tts_language,
                    tts_instruct=tts_instruct,
                )
            else:
                if not text_input:
                    return "请输入食材列表或选择图片上传", "", None
                result = recipe_recommender_pipeline(
                    model_manager=model_manager,
                    input_text=text_input,
                    input_type="text",
                    vlm_max_new_tokens=vlm_max_new_tokens,
                    tts_max_new_tokens=tts_max_new_tokens,
                    tts_speaker=tts_speaker,
                    tts_language=tts_language,
                    tts_instruct=tts_instruct,
                )

            ingredients = result.get("ingredients", "")
            recipe = result.get("recipe", "")

            # Save audio as temp file if exists
            audio_path = None
            if enable_tts and result.get("audio") is not None:
                sr, wav_data = result["audio"]
                audio_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                write(audio_tmp.name, sr, wav_data.astype(np.float32))
                audio_path = audio_tmp.name

            return ingredients, recipe, audio_path
        finally:
            if tmp_path:
                import os
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # Get TTS supported speakers and languages (lazy load TTS model)
    supported_speakers, supported_languages = model_manager.get_tts_speakers_and_languages()
    model_manager.release_tts()

    with gr.Blocks(title="智能食谱推荐系统") as demo:
        gr.Markdown("# 🍳 智能食谱推荐系统")
        gr.Markdown("上传食材图片或输入食材清单，系统将识别食材并生成个性化菜谱，可选择语音播报。")

        with gr.Row():
            with gr.Column(scale=1):
                input_type = gr.Radio(["文本输入", "图片上传"], label="输入方式", value="文本输入")

                text_input = gr.Textbox(label="请输入食材（逗号分隔）", placeholder="例如：番茄, 鸡蛋, 洋葱", visible=True)
                image_input = gr.Image(label="上传食材图片", type="filepath", visible=False)

                with gr.Accordion("语音合成设置", open=True):
                    enable_tts = gr.Checkbox(label="启用语音播报", value=False)
                    tts_speaker = gr.Dropdown(choices=supported_speakers, value=supported_speakers[0], label="说话人")
                    tts_language = gr.Dropdown(choices=supported_languages, value=supported_languages[0], label="语言")
                    tts_instruct = gr.Textbox(value="用友好亲切的语气说话。", label="风格指令")

                run_btn = gr.Button("生成食谱", variant="primary")

            with gr.Column(scale=1):
                ingredients_output = gr.Textbox(label="识别到的食材", lines=8)
                recipe_output = gr.Textbox(label="推荐食谱", lines=20)
                audio_output = gr.Audio(label="语音指导", type="filepath")

        def toggle_inputs(choice):
            return [gr.update(visible=choice == "文本输入"), gr.update(visible=choice == "图片上传")]

        input_type.change(toggle_inputs, inputs=input_type, outputs=[text_input, image_input])

        run_btn.click(
            gradio_pipeline,
            inputs=[input_type, text_input, image_input, enable_tts, tts_speaker, tts_language, tts_instruct],
            outputs=[ingredients_output, recipe_output, audio_output],
        )

    return demo
