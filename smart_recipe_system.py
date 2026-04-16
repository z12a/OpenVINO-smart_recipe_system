 # -*- coding: utf-8 -*-
"""
智能食谱推荐系统 (基于 Qwen 大模型)
 
"""

import json
from typing import Dict, Optional

class SmartRecipeSystem:
    def __init__(self, model=None):
        self.model = model
        print("智能食谱系统已加载 ✅")

    def generate_recipe(self, user_input: str, max_tokens: int = 1024) -> Dict:
        """
        核心生成函数：模仿文档中的 transcribe 逻辑，改为 generate
        Args:
            user_input (str): 用户输入，例如 "我想吃番茄和鸡蛋" 或 "推荐一个川菜"
            max_tokens (int): 生成的最大长度
        Returns:
            dict: 包含菜名、食材、步骤的食谱
        """
        
        # 1. 构建 Prompt (类似于文档中 ASR 转写后的文本处理)
        # 我们参考文档中对模型的控制方式，要求结构化输出
        system_prompt = (
            "你是一位世界顶级的中餐大厨。"
            "请根据用户提供的食材或菜名，生成一份详细的食谱。"
            "请严格遵守以下格式输出：\n"
            "【菜名】: (生成一个吸引人的名字)\n"
            "【简介】: (一句话描述这道菜的特点)\n"
            "【所需食材】: (列出主要食材和调料，包含大致分量)\n"
            "【烹饪步骤】: (分步骤列出烹饪过程，每步一行)\n"
            "【小贴士】: (提供一个烹饪技巧或替代食材建议)"
        )
        
        # 2. 模拟文档中的模型调用流程
        # 在文档中是 ov_model.transcribe(...)，这里我们改为生成文本
        try:
            # 模拟文档中的推理参数
            results = self._mock_model_inference(
                user_input=user_input,
                text_input=f"请为以下需求生成食谱: {user_input}",
                prompt=system_prompt,
                max_new_tokens=max_tokens
            )
            
            # 3. 模拟后处理 (模仿文档中对 results[0].text 的处理)
            raw_text = results[0].text if isinstance(results, list) else results.text
            
            # 4. 解析食谱输出
            recipe = self._parse_recipe(raw_text)
            return recipe
            
        except Exception as e:
            print(f"生成食谱时出错: {e}")
            return {"error": str(e)}

    def _mock_model_inference(self, user_input: str, text_input: str, prompt: str, max_new_tokens: int):
        """
        模拟 Qwen 模型推理（用于演示）
        实际应用中请替换为真实的 LLM 调用
        """
        # 模拟不同输入的食谱生成结果
        mock_recipes = {
            "番茄和鸡蛋": """【菜名】: 番茄炒蛋
【简介】: 经典家常下饭菜，酸甜可口，营养丰富
【所需食材】: 番茄2个、鸡蛋3个、葱花适量、盐1小勺、糖1小勺、生抽1勺、食用油2勺
【烹饪步骤】: 1. 鸡蛋打散，加少许盐调味；2. 番茄切块备用；3. 热锅倒油，倒入鸡蛋液炒至凝固盛出；4. 锅中再加少许油，放入番茄块翻炒出汁；5. 加入炒好的鸡蛋，加糖、盐、生抽调味；6. 翻炒均匀后撒葱花出锅
【小贴士】: 番茄选择熟透的更出汁，炒鸡蛋时火不要太大以免炒老""",
            
            "土豆牛肉胡萝卜炖菜": """【菜名】: 土豆胡萝卜炖牛肉
【简介】: 冬日暖身滋补菜，肉质软烂，汤汁浓郁
【所需食材】: 牛腩500克、土豆2个、胡萝卜1根、姜片3片、葱段1段、料酒2勺、生抽2勺、老抽1勺、冰糖5颗、八角2颗、香叶2片、盐适量
【烹饪步骤】: 1. 牛腩切块焯水，去除血沫；2. 锅中倒油，爆香姜片葱段；3. 加入牛腩翻炒，加料酒、生抽、老抽上色；4. 加入开水没过牛肉，放入八角、香叶、冰糖；5. 小火慢炖1小时；6. 加入土豆胡萝卜块继续炖30分钟；7. 加盐调味即可
【小贴士】: 牛腩选择带筋的更有嚼劲，炖的时候可以加少量醋帮助肉质软化""",
            
            "辣的快手菜": """【菜名】: 酸辣土豆丝
【简介】: 开胃下饭快手菜，酸辣爽脆，5分钟搞定
【所需食材】: 土豆2个、干辣椒3-4个、蒜2瓣、醋2勺、盐1小勺、糖少许、食用油2勺
【烹饪步骤】: 1. 土豆切细丝泡水去淀粉；2. 蒜切末，干辣椒切段；3. 热锅热油，爆香蒜末和干辣椒；4. 倒入土豆丝快速翻炒；5. 加醋、盐、糖调味；6. 翻炒至土豆丝变透明即可出锅
【小贴士】: 土豆丝不要炒太久保持脆感，醋要后放才能保留酸味""",
            
            "低卡沙拉": """【菜名】: 鸡胸肉蔬菜沙拉
【简介】: 减脂期必备，高蛋白低热量，清爽可口
【所需食材】: 鸡胸肉1块、生菜1颗、番茄1个、黄瓜半根、玉米粒50克、鸡蛋1个、橄榄油1勺、柠檬汁1勺、黑胡椒适量、盐少许
【烹饪步骤】: 1. 鸡胸肉煮熟切块；2. 鸡蛋煮熟切块；3. 生菜撕片，番茄黄瓜切块；4. 所有食材放入碗中；5. 橄榄油、柠檬汁、盐、黑胡椒混合拌匀；6. 淋在沙拉上即可
【小贴士】: 可以加入牛油果增加健康脂肪，酱汁不要太多控制热量"""
        }
        
        # 根据用户输入匹配食谱
        # 使用更灵活的匹配方式，检查用户输入中是否包含关键词的主要部分
        keyword_mappings = {
            "土豆牛肉胡萝卜炖菜": ["土豆", "牛肉", "胡萝卜", "炖菜"],
            "番茄和鸡蛋": ["番茄", "鸡蛋"],
            "辣的快手菜": ["辣", "快手", "川菜"],
            "低卡沙拉": ["低卡", "沙拉", "减肥"]
        }
        
        for recipe_keyword, keywords in keyword_mappings.items():
            if any(keyword in user_input for keyword in keywords):
                return [type('obj', (object,), {'text': mock_recipes[recipe_keyword]})]
        
        # 默认食谱
        default_recipe = f"""【菜名】: 美味家常菜
【简介】: 根据您的口味精心调配的家常美食
【所需食材】: 根据您的需求定制
【烹饪步骤】: 请提供更具体的食材，我将为您生成详细步骤
【小贴士】: 烹饪时注意火候控制，保持食材新鲜"""
        return [type('obj', (object,), {'text': default_recipe})]

    def _parse_recipe(self, text: str) -> Dict:
        """
        解析模型输出的文本为结构化数据
        """
        recipe = {
            "菜名": "",
            "简介": "",
            "所需食材": "",
            "烹饪步骤": "",
            "小贴士": "",
            "raw_output": text
        }
        
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            if "【菜名】" in line:
                recipe["菜名"] = line.replace("【菜名】", "").strip(": ").strip()
            elif "【简介】" in line:
                recipe["简介"] = line.replace("【简介】", "").strip(": ").strip()
            elif "【所需食材】" in line:
                recipe["所需食材"] = line.replace("【所需食材】", "").strip(": ").strip()
            elif "【烹饪步骤】" in line:
                recipe["烹饪步骤"] = line.replace("【烹饪步骤】", "").strip(": ").strip()
            elif "【小贴士】" in line:
                recipe["小贴士"] = line.replace("【小贴士】", "").strip(": ").strip()
            elif recipe["烹饪步骤"] and not any(section in line for section in ["【菜名】", "【简介】", "【所需食材】", "【烹饪步骤】", "【小贴士】"]):
                recipe["烹饪步骤"] += "\n" + line.strip()
        
        return recipe

# --------------------------------------------------
# 2. 交互式演示 (模仿文档中的 "交互式演示" 章节)
# --------------------------------------------------

def interactive_demo():
    print("# 智能食谱推荐系统")
    print("基于 Qwen 大模型为您定制专属食谱\n")
    
    # --- 模拟文档中的模型加载流程 ---
    # 文档中: ov_model = OVQwen3ASRModel.from_pretrained(...)
    print("正在加载 Qwen 模型...")
    try:
        # 这里我们假设模型已经加载好，或者使用一个 Mock 对象
        ov_model = "Mock_OVQwen_Model"
        
        recipe_system = SmartRecipeSystem(ov_model)
        
        # --- 模仿文档中的测试音频步骤，改为测试食谱 ---
        test_inputs = [
            "我有土豆、牛肉和胡萝卜，想吃炖菜",
            "我很饿，想吃辣，推荐一个快手菜",
            "我正在减肥，推荐一个低卡的沙拉食谱"
        ]
        
        for i, dish in enumerate(test_inputs, 1):
            print(f"\n--- 测试 {i}: 用户需求 - {dish} ---")
            print("正在思考烹饪步骤...")
            
            # 调用我们的生成函数
            recipe = recipe_system.generate_recipe(dish)
            
            # --- 模仿文档中的打印结果 ---
            if "error" not in recipe:
                print(f"\n✅ 推荐菜名: {recipe.get('菜名', '未命名美食')}")
                print(f"📖 简介: {recipe.get('简介', '')}")
                print(f"\n🥬 所需食材:\n{recipe.get('所需食材', '')}")
                print(f"\n👩🍳 烹饪步骤:\n{recipe.get('烹饪步骤', '')}")
                print(f"\n💡 小贴士: {recipe.get('小贴士', '')}")
            else:
                print(f"❌ 错误: {recipe['error']}")
                
    except Exception as e:
        print(f"初始化失败，请检查模型环境: {e}")
        print("提示: 本代码基于 lab5.ipynb 架构重构，需配合 Qwen 模型运行。")

# --------------------------------------------------
# 3. 运行系统
# --------------------------------------------------

if __name__ == "__main__":
    interactive_demo()
