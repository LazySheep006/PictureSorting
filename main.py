import os
import gc
import math
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from tqdm import tqdm

# 1. 参数配置

SOURCE_DIR = 'images'                 # 素材文件夹
REF_IMAGE_PATH = 'auto_heart_ref.jpg' # 蓝图
OUTPUT_NAME = 'final_spatial_mosaic.jpg'

OUTPUT_WIDTH = 3000                   # 高清
GRID_COLS = 60                        # 60 列，高密度！
TILE_ASPECT = 3/4                     # 竖构图

# 后期遮罩强度 (0.15 - 0.2 比较合适，淡淡的一层)
MASK_INTENSITY = 0.00 #个人确定吧，我自己的图片不需要也能看得出心形的

# 【核心】空间排斥半径 (单位: 格子)
# 意思是：一旦某张图用过了，它周围 R 个格子范围内，绝不允许再出现这张图
# 60列的话，设为 12-15 比较合适，保证一张图在局部视野里只出现一次
SPATIAL_RADIUS = 12

# 2. 逻辑实现

def load_images_smart(source_dir, tile_w, tile_h):
    tiles = []
    avg_rgbs = []
    
    valid = {'.jpg', '.png', '.jpeg', '.bmp', '.webp'}
    files = [f for f in os.listdir(source_dir) if os.path.splitext(f)[1].lower() in valid]
    np.random.shuffle(files) # 必须打乱
    
    if len(files) == 0: raise Exception("images 文件夹空的！")
    print(f"加载素材: {len(files)} 张")
    
    # 稍微读大一点，抗锯齿
    load_w, load_h = int(tile_w * 1.5), int(tile_h * 1.5)

    for f in tqdm(files, desc="预处理"):
        try:
            path = os.path.join(source_dir, f)
            with Image.open(path) as img:
                img = img.convert('RGB')
                img.thumbnail((load_w, load_h), Image.BICUBIC)
                tile = ImageOps.fit(img, (tile_w, tile_h), method=Image.LANCZOS)
                c = np.array(tile.resize((1, 1)).getpixel((0, 0)))
                tiles.append(tile)
                avg_rgbs.append(c)
        except:
            continue
    return tiles, np.array(avg_rgbs)

def apply_subtle_mask(canvas, ref_path, intensity=0.2):
    """后期加上淡淡的遮罩"""
    print(f"\n应用微弱遮罩 (强度 {intensity})...")
    original = canvas.copy()
    darkener = ImageEnhance.Brightness(original)
    # 变暗一点点
    darkened = darkener.enhance(1.0 - intensity)
    
    mask = Image.open(ref_path).convert('L')
    mask = mask.resize(canvas.size, Image.BICUBIC)
    # 模糊遮罩边缘，让过渡自然
    mask = mask.filter(ImageFilter.GaussianBlur(radius=20))
    
    return Image.composite(original, darkened, mask)

def main():
    # --- A. 算网格 ---
    tile_w = OUTPUT_WIDTH // GRID_COLS
    tile_h = int(tile_w / TILE_ASPECT)
    
    if not os.path.exists(REF_IMAGE_PATH): return
    ref_img = Image.open(REF_IMAGE_PATH).convert('RGB')
    rw, rh = ref_img.size
    grid_rows = int((rh / rw) * GRID_COLS * (tile_w / tile_h))
    
    total_slots = GRID_COLS * grid_rows
    print(f"高密网格: {GRID_COLS}x{grid_rows} = {total_slots} 格")
    
    # 缩小参考图
    ref_small = ref_img.resize((GRID_COLS, grid_rows), Image.LANCZOS)
    target_pixels = np.array(ref_small)

    # 加载素材
    tiles, source_rgbs = load_images_smart(SOURCE_DIR, tile_w, tile_h)
    num_imgs = len(tiles)
    print(f"{num_imgs} 张图填满 {total_slots} 格 (平均每张用 {total_slots//num_imgs} 次)")
    print(f"同一张图必须间隔 {SPATIAL_RADIUS} 个格子以上")

    # --- B. 拼图 ---
    canvas = Image.new('RGB', (OUTPUT_WIDTH, grid_rows * tile_h))
    
    # 记录每张图都用在了哪些位置: {img_id: [(r,c), (r,c)...]}
    usage_history = {i: [] for i in range(num_imgs)}
    # 记录总使用次数
    usage_counts = np.zeros(num_imgs, dtype=int)
    
    # 候选池不用太大，因为我们主要靠空间规则筛选
    pool_size = min(300, num_imgs)

    for r in tqdm(range(grid_rows), desc="排兵布阵"):
        for c in range(GRID_COLS):
            target_rgb = target_pixels[r, c]
            # 1. 颜色距离
            diff = source_rgbs - target_rgb
            dists = np.sum(diff**2, axis=1)    
            # 2. 拿前 N 个颜色最像的
            candidates = np.argpartition(dists, pool_size-1)[:pool_size]    
            best_idx = -1
            min_score = float('inf')
            
            # 3. 遍历候选者，寻找最佳 (颜色好 + 次数少 + 距离远)
            for idx in candidates:
                # 空间互斥检查
                is_safe = True
                # 检查这张图以前出现过的所有位置
                # 为了性能，我们倒序检查（最近用过的肯定在列表后面）
                for (pr, pc) in reversed(usage_history[idx]):
                    # 欧氏距离检查: sqrt((r1-r2)^2 + (c1-c2)^2)
                    dist_sq = (r - pr)**2 + (c - pc)**2
                    if dist_sq < SPATIAL_RADIUS**2:
                        is_safe = False
                        break
                    # 如果行距已经超过半径了，就不用往前查
                    if (r - pr) > SPATIAL_RADIUS:
                        break
                if not is_safe:
                    continue # 距离太近，跳过，看下一个候选者
                # 依然优先使用次数少的
                score = dists[idx] + (usage_counts[idx] * 500000)
                
                if score < min_score:
                    min_score = score
                    best_idx = idx
            
            # 4. 保底机制 (如果前300个都离得太近怎么办？)
            if best_idx == -1:
                # 这是一个紧急情况。我们被迫缩小安全半径，在全库里找一个
                # 或者直接找一个距离最远的
                # 简单处理：找一个颜色最像的，且不是直接紧邻的(半径=1.5)
                top_sorted = candidates[np.argsort(dists[candidates])]
                for idx in top_sorted:
                    # 只要不是紧邻 (左、上、左上、右上)
                    last_pos = usage_history[idx][-1] if usage_history[idx] else (-99,-99)
                    if abs(r - last_pos[0]) > 1 or abs(c - last_pos[1]) > 1:
                        best_idx = idx
                        break
                if best_idx == -1: best_idx = top_sorted[0]

            # 5. 落子
            usage_history[best_idx].append((r, c))
            usage_counts[best_idx] += 1
            
            x = c * tile_w
            y = r * tile_h
            canvas.paste(tiles[best_idx], (x, y))
            
        if r % 5 == 0: gc.collect()

    # --- C. 后期与保存 ---
    print(f"\n最多的一张图用了 {np.max(usage_counts)} 次，最少的用了 {np.min(usage_counts)} 次")
    
    final_canvas = apply_subtle_mask(canvas, REF_IMAGE_PATH, intensity=MASK_INTENSITY)
    
    print(f"保存: {OUTPUT_NAME}")
    final_canvas.save(OUTPUT_NAME, quality=95, subsampling=0)
    print("完成！")
if __name__ == '__main__':
    main()