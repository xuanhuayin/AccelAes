#!/usr/bin/env python3
"""
从 Pick-a-Pic 完整数据集中自动提取美学/质量/风格锚点短语。

策略：
  1. 读取全部 ~58万条 prompt 的 caption
  2. 提取 unigram + bigram + trigram
  3. 用预定义美学类别种子模式过滤
  4. 按频率排序，按类别分组
  5. 输出推荐锚点列表
"""

import json
import re
import sys
import os
from collections import Counter

# ---- 美学/质量相关的正则模式 ----
# 每个模式对应一个语义类别
CATEGORY_PATTERNS = {
    "quality": (
        r"\b("
        r"high quality|low quality|best quality|ultra quality|"
        r"high res(?:olution)?|ultra high resolution|"
        r"ultra detailed|insanely detailed|extremely detailed|"
        r"highly detailed|intricately detailed|finely detailed|"
        r"super detailed|very detailed|incredibly detailed|"
        r"hyper ?detailed|hyper ?realistic|photo ?realistic|ultra ?realistic|"
        r"8k|4k|16k|32k|64k|128k|12k|"
        r"hdr|uhd|dslr|raw photo"
        r")\b"
    ),
    "style": (
        r"\b("
        r"oil painting|digital art|digital painting|digital illustration|"
        r"concept art|splash art|watercolor|pencil drawing|charcoal drawing|"
        r"ink drawing|line ?art|anime style|manga style|cartoon style|"
        r"illustration style|photography|photograph|cinematic|filmic|"
        r"octane render|unreal engine|ray tracing|3d render|cgi|"
        r"impressionist|surrealist|art nouveau|art deco|"
        r"gothic style|steampunk|cyberpunk|fantasy art|"
        r"pop surrealism|studio ghibli|pixel art|vector art|"
        r"watercolor painting|acrylic painting|pastel drawing|"
        r"mixed media|collage|photomanipulation|matte painting|"
        r"concept art portrait|fantasy illustration"
        r")\b"
    ),
    "aesthetic": (
        r"\b("
        r"beautiful|gorgeous|elegant|stunning|magnificent|majestic|"
        r"exquisite|ornate|intricate|elaborate|vivid|vibrant|"
        r"dramatic|epic|inspiring|ethereal|mystical|magical|enchanting|"
        r"dark|grim|gritty|moody|atmospheric|dreamy|whimsical|"
        r"cute|kawaii|adorable|charming|delicate|graceful|"
        r"surreal|abstract|minimalist|luxurious|opulent"
        r")\b"
    ),
    "lighting": (
        r"\b("
        r"cinematic lighting|volumetric lighting|natural light(?:ing)?|"
        r"studio light(?:ing)?|dramatic lighting|dynamic light(?:ing)?|"
        r"global illumination|ambient lighting|rim light(?:ing)?|"
        r"backlight(?:ing)?|golden hour|blue hour|neon light(?:ing)?|"
        r"soft lighting|hard lighting|chiaroscuro|rembrandt lighting|"
        r"bioluminescen(?:t|ce)|god ?rays|light rays|"
        r"subsurface scattering|caustics"
        r")\b"
    ),
    "color": (
        r"\b("
        r"muted colors|pastel colors|warm colors|cool colors|"
        r"vivid colors|bright colors|vibrant colors|rich colors|"
        r"color grading|technicolor|triadic colors|complementary colors|"
        r"monochrome|sepia|desaturated|saturated|"
        r"black and white|color palette|color scheme|"
        r"iridescent|chromatic|prismatic"
        r")\b"
    ),
    "detail_focus": (
        r"\b("
        r"sharp focus|soft focus|depth of field|dof|bokeh|"
        r"close ?up|closeup|wide angle|aerial view|"
        r"full body|portrait|headshot|bust shot|"
        r"detailed|realistic|sharp|smooth|crisp|clean lines|"
        r"complex|fine detail|ultra ?fine|"
        r"macro|micro detail|texture|textured|"
        r"intricate details?|intricate environment|"
        r"extreme detail|extreme close ?up"
        r")\b"
    ),
    "composition": (
        r"\b("
        r"centered|symmetrical?|rule of thirds|"
        r"foreground|background|main subject|focal point|"
        r"leading lines|negative space|framing|"
        r"bird(?:'s)? eye view|worm(?:'s)? eye view|"
        r"isometric|panoramic|wide shot|medium shot|"
        r"over the shoulder|dutch angle"
        r")\b"
    ),
    "platform_quality": (
        r"\b("
        r"trending on artstation|artstation|deviantart|pixiv|"
        r"award[- ]?winning|masterpiece|professional|"
        r"museum quality|gallery quality|"
        r"cover art|key visual|splash page|"
        r"by greg rutkowski|by artgerm|by wlop|by alphonse mucha|"
        r"by loish|by rossdraws"
        r")\b"
    ),
}


def extract_anchors(meta_path, max_lines=None):
    """从 meta.jsonl 中提取锚点短语"""

    # 按类别统计
    cat_counters = {cat: Counter() for cat in CATEGORY_PATTERNS}
    global_counter = Counter()

    total_prompts = 0
    unique_prompts = set()

    with open(meta_path, "r") as f:
        for line_num, line in enumerate(f):
            if max_lines and line_num >= max_lines:
                break

            try:
                data = json.loads(line.strip())
                caption = data.get("caption", "")
            except json.JSONDecodeError:
                continue

            if not caption or caption in unique_prompts:
                continue
            unique_prompts.add(caption)
            total_prompts += 1

            caption_lower = caption.lower()

            for cat_name, pattern in CATEGORY_PATTERNS.items():
                matches = re.findall(pattern, caption_lower, re.IGNORECASE)
                for m in matches:
                    m = m.strip().lower()
                    cat_counters[cat_name][m] += 1
                    global_counter[m] += 1

            if total_prompts % 50000 == 0:
                print(f"  已处理 {total_prompts} 条唯一 prompt...", file=sys.stderr)

    return total_prompts, cat_counters, global_counter


def main():
    meta_path = "/home/runkai/xuanhua/pickapic_v1/meta.jsonl"

    print(f"从 {meta_path} 提取锚点短语...\n", file=sys.stderr)
    total, cat_counters, global_counter = extract_anchors(meta_path)

    print(f"共处理 {total} 条唯一 prompt\n")

    # ---- 按类别输出 ----
    print("=" * 70)
    print("【按类别分组统计】")
    print("=" * 70)

    all_recommended = []

    for cat_name, counter in cat_counters.items():
        if not counter:
            continue
        items = counter.most_common(30)
        total_freq = sum(c for _, c in counter.items())
        print(f"\n  [{cat_name}] ({len(counter)} 个不同短语, 总出现 {total_freq} 次)")
        print(f"  {'—' * 50}")
        for phrase, count in items:
            pct = count / total * 100
            print(f"    {count:6d}x ({pct:5.2f}%)  {phrase}")

    # ---- 全局排名 ----
    print("\n" + "=" * 70)
    print("【全局频率 Top 60】")
    print("=" * 70)
    for i, (phrase, count) in enumerate(global_counter.most_common(60)):
        pct = count / total * 100
        print(f"  {i+1:3d}. {count:6d}x ({pct:5.2f}%)  \"{phrase}\"")

    # ---- 推荐锚点：每个类别取 top 若干，确保覆盖面 ----
    print("\n" + "=" * 70)
    print("【推荐锚点列表】每类别取 top-N，freq >= 500")
    print("=" * 70)

    # 每类取多少个
    top_n_per_cat = {
        "quality": 8,
        "style": 8,
        "aesthetic": 10,
        "lighting": 6,
        "color": 4,
        "detail_focus": 8,
        "composition": 4,
        "platform_quality": 6,
    }

    recommended = []
    seen = set()
    for cat_name, counter in cat_counters.items():
        n = top_n_per_cat.get(cat_name, 5)
        for phrase, count in counter.most_common(n):
            if count >= 500 and phrase not in seen:
                recommended.append((phrase, count, cat_name))
                seen.add(phrase)

    # 按频率排序
    recommended.sort(key=lambda x: -x[1])

    for i, (phrase, count, cat) in enumerate(recommended):
        pct = count / total * 100
        print(f"  {i+1:2d}. [{cat:18s}] {count:6d}x ({pct:5.2f}%)  \"{phrase}\"")

    print(f"\n共推荐 {len(recommended)} 个锚点短语")

    # ---- 输出 Python 代码 ----
    print("\n" + "=" * 70)
    print("【Python 代码】可直接复制到 SemanticMaskBuilder.DEFAULT_ANCHORS")
    print("=" * 70)
    print("DEFAULT_ANCHORS = [")
    for phrase, count, cat in recommended:
        print(f'    "{phrase}",  # {cat}, {count}x ({count/total*100:.1f}%)')
    print("]")

    # ---- 也输出一个低门槛版本 (freq >= 200) ----
    print("\n# ---- 扩展版 (freq >= 200) ----")
    extended = []
    seen2 = set()
    for cat_name, counter in cat_counters.items():
        for phrase, count in counter.most_common(15):
            if count >= 200 and phrase not in seen2:
                extended.append((phrase, count, cat_name))
                seen2.add(phrase)
    extended.sort(key=lambda x: -x[1])

    print(f"EXTENDED_ANCHORS = [  # {len(extended)} phrases")
    for phrase, count, cat in extended:
        print(f'    "{phrase}",  # {cat}, {count}x')
    print("]")


if __name__ == "__main__":
    main()
