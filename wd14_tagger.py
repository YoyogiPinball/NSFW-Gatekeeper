import os
import re
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from datetime import datetime

# 設定
INPUT_DIR = "origin"  # 処理する画像フォルダ
NSFW_DIR = "_moto/nsfw"  # NSFW画像フォルダ（手動振り分け）
SFW_DIR = "_moto/sfw"    # SFW画像フォルダ（手動振り分け）
LOG_DIR = "logs"  # ログ出力フォルダ
MODEL_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
BATCH_SIZE = 8  # バッチサイズ（メモリに応じて調整）

# NSFW判定ルール（優先度順）

# ルール1: これらのタグがこの値以上ならNSFW確定
NSFW_RULES = {
    # 性器露出（高優先度）
    'penis': 0.4,              # ペニスが見える
    'pussy': 0.4,              # 性器が見える
    'clitoris': 0.3,           # クリトリス
    'genitals': 0.4,           # 性器全般
    'uncensored': 0.5,         # 無修正

    # 乳首・乳輪露出
    'nipples': 0.5,            # 乳首が見える
    'areolae': 0.4,            # 乳輪が見える
    'exposed_nipples': 0.4,    # 乳首露出

    # 性行為
    'sex': 0.6,                # 性行為
    'vaginal': 0.6,            # 挿入
    'oral': 0.6,               # 口淫
    'anal': 0.6,               # アナル
    'paizuri': 0.5,            # パイズリ
    'fellatio': 0.5,           # フェラチオ
    'cunnilingus': 0.5,        # クンニリングス
    'masturbation': 0.6,       # 自慰
    'penetration': 0.5,        # 挿入行為

    # その他
    'cum': 0.5,                # 精液
    'spread_pussy': 0.4,       # 性器を広げている
}

# ルール2: これらのタグが全てこの値未満ならSFW確定の候補
SFW_CHECK_TAGS = [
    'nipples', 'penis', 'pussy', 'sex', 'vaginal', 'oral',
    'areolae', 'genitals', 'exposed_nipples', 'clitoris',
    'paizuri', 'fellatio', 'cunnilingus', 'masturbation'
]
SFW_MAX_THRESHOLD = 0.25  # 閾値を少し厳しく

# ルール3（フォールバック）: explicitスコアによる複合判定
EXPLICIT_HIGH = 0.30  # この値以上は確実にNSFW（厳しく調整）
EXPLICIT_LOW = 0.10   # この値未満は確実にSFW（厳しく調整）
NSFW_TAG_SUM_THRESHOLD = 0.068  # 中間帯でのNSFWタグ合計閾値
EXPLICIT_MAX_FOR_SFW = 0.20  # ルール3でSFW判定する場合のexplicit上限

# WD14モデルのロード
print("モデルをダウンロード中...")
model_path = hf_hub_download(MODEL_REPO, "model.onnx")
label_path = hf_hub_download(MODEL_REPO, "selected_tags.csv")

print("モデルをロード中...")
session = ort.InferenceSession(
    model_path,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# ラベル読み込み
import csv
with open(label_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    tags = [row for row in reader]

# rating関連のタグを探す
rating_tags = {}
for i, tag in enumerate(tags):
    if tag['category'] == '9':  # ratingカテゴリ
        rating_tags[tag['name']] = i

# NSFW判定用の性的タグ（詳細版）
nsfw_tags = {}
nsfw_tag_names = [
    # 基本的な性的タグ
    'nipples', 'pussy', 'penis', 'anus', 'testicles', 'cum', 'sex', 'vaginal', 'oral', 'anal',
    # 詳細な性器・露出タグ
    'areolae', 'nude', 'completely_nude', 'clitoris', 'genitals',
    'exposed_nipples', 'female_pubic_hair', 'male_pubic_hair', 'pubic_hair',
    'erection', 'uncensored', 'spread_legs', 'spread_pussy',
    # 性行為関連
    'masturbation', 'paizuri', 'fellatio', 'cunnilingus', 'penetration'
]
for i, tag in enumerate(tags):
    if tag['name'] in nsfw_tag_names:
        nsfw_tags[tag['name']] = i

print(f"Rating tags found: {list(rating_tags.keys())}")
print(f"NSFW tags found: {len(nsfw_tags.keys())}個のタグ")
print(f"\n判定ルール（優先度順）:")
print(f"  ルール0: タグ組み合わせパターン検出（性器露出・乳首露出・性行為など）")
print(f"  ルール1: 個別タグ閾値チェック（{len(NSFW_RULES)}個のタグ）")
print(f"  ルール2: explicitスコア複合判定 (High={EXPLICIT_HIGH}, Low={EXPLICIT_LOW})")
print(f"  ルール3: 全NSFWタグ < {SFW_MAX_THRESHOLD} AND explicit < {EXPLICIT_MAX_FOR_SFW} → SFW")
print(f"  デフォルト: 該当なし → NSFW")

def preprocess_image(image_path):
    """画像を前処理"""
    img = Image.open(image_path).convert('RGB')
    # WD14の入力サイズは448x448
    img = img.resize((448, 448), Image.LANCZOS)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, 0)  # バッチ次元追加
    return img_array

def get_rating(image_path):
    """画像のrating判定とNSFWタグ取得"""
    img_array = preprocess_image(image_path)

    # 推論
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_array})[0][0]

    # ratingスコア取得
    ratings = {}
    for name, idx in rating_tags.items():
        ratings[name] = float(output[idx])

    # NSFWタグスコア取得
    nsfw_scores = {}
    for name, idx in nsfw_tags.items():
        nsfw_scores[name] = float(output[idx])

    # 最高スコアのratingを返す
    best_rating = max(ratings.items(), key=lambda x: x[1])[0]
    return best_rating, ratings, nsfw_scores

def determine_sfw_nsfw(ratings, nsfw_scores):
    """SFW/NSFW判定（ルールベース + パターンマッチング）"""
    explicit_score = ratings.get('explicit', 0.0)

    # ルール0: タグ組み合わせパターンでNSFW確定（最優先）
    # パターン1: 性器が明確に見える
    genitals_visible = (
        nsfw_scores.get('penis', 0) >= 0.3 or
        nsfw_scores.get('pussy', 0) >= 0.3 or
        nsfw_scores.get('genitals', 0) >= 0.3 or
        nsfw_scores.get('anus', 0) >= 0.3
    )
    if genitals_visible:
        return 'N', f"ルール0: 性器露出パターン検出"

    # パターン2: 乳首・乳輪露出
    nipple_exposed = (
        nsfw_scores.get('nipples', 0) >= 0.3 or
        nsfw_scores.get('areolae', 0) >= 0.3 or
        nsfw_scores.get('exposed_nipples', 0) >= 0.3
    )
    if nipple_exposed:
        return 'N', f"ルール0: 乳首露出パターン検出"

    # パターン3: 性行為の組み合わせ
    sex_act = (
        (nsfw_scores.get('sex', 0) >= 0.4) or
        (nsfw_scores.get('paizuri', 0) >= 0.3) or
        (nsfw_scores.get('fellatio', 0) >= 0.3) or
        (nsfw_scores.get('cunnilingus', 0) >= 0.3) or
        (nsfw_scores.get('vaginal', 0) >= 0.4 and nsfw_scores.get('penis', 0) >= 0.2)
    )
    if sex_act:
        return 'N', f"ルール0: 性行為パターン検出"

    # パターン4: 全裸 + 性的要素
    nude_sexual = (
        (nsfw_scores.get('nude', 0) >= 0.5 or nsfw_scores.get('completely_nude', 0) >= 0.5) and
        (nsfw_scores.get('nipples', 0) >= 0.2 or nsfw_scores.get('pussy', 0) >= 0.2 or nsfw_scores.get('penis', 0) >= 0.2)
    )
    if nude_sexual:
        return 'N', f"ルール0: 全裸+性的要素パターン検出"

    # ルール1: 個別タグでNSFW確定チェック
    for tag_name, threshold in NSFW_RULES.items():
        tag_score = nsfw_scores.get(tag_name, 0.0)
        if tag_score >= threshold:
            return 'N', f"ルール1: {tag_name}:{tag_score:.3f} >= {threshold}"

    # ルール2: explicitスコアによる複合判定（優先度UP）
    if explicit_score >= EXPLICIT_HIGH:
        return 'N', f"ルール2: explicit:{explicit_score:.3f} >= {EXPLICIT_HIGH}"
    elif explicit_score < EXPLICIT_LOW:
        return 'S', f"ルール2: explicit:{explicit_score:.3f} < {EXPLICIT_LOW}"
    else:
        # 中間帯：NSFWタグの合計で判定
        nsfw_sum = (nsfw_scores.get('nipples', 0.0) +
                    nsfw_scores.get('penis', 0.0) +
                    nsfw_scores.get('pussy', 0.0))
        if nsfw_sum > NSFW_TAG_SUM_THRESHOLD:
            return 'N', f"ルール2: タグ合計:{nsfw_sum:.3f} > {NSFW_TAG_SUM_THRESHOLD}"
        # ルール3へ続く

    # ルール3: 全NSFWタグが低く、かつexplicitも低ければSFW確定（最後の判定）
    all_low = all(nsfw_scores.get(tag, 0.0) < SFW_MAX_THRESHOLD for tag in SFW_CHECK_TAGS)
    if all_low and explicit_score < EXPLICIT_MAX_FOR_SFW:
        return 'S', f"ルール3: 全NSFWタグ < {SFW_MAX_THRESHOLD} AND explicit:{explicit_score:.3f} < {EXPLICIT_MAX_FOR_SFW}"

    # どのルールにも該当しない場合はNSFW（安全側に倒す）
    return 'N', f"デフォルト: 該当ルールなし（NSFW側に倒す）"

def add_sfw_nsfw_tag(filename, tag):
    """ファイル名にS/Nタグを追加"""
    # {zpi$t=...}を探す
    pattern = r'\{zpi\$t=([^}]+)\}'
    match = re.search(pattern, filename)

    if not match:
        # タグがない場合は新規作成
        # 拡張子を取得
        name_parts = filename.rsplit('.', 1)
        if len(name_parts) == 2:
            base_name, ext = name_parts
            new_filename = f"{base_name}{{zpi$t={tag}}}.{ext}"
        else:
            # 拡張子がない場合（稀だが）
            new_filename = f"{filename}{{zpi$t={tag}}}"
        return new_filename

    current_tags = match.group(1)

    # 既にS or Nが含まれているかチェック
    if 'S' in current_tags or 'N' in current_tags:
        return None  # スキップ

    # 新しいタグを追加
    new_tags = f"{current_tags},{tag}"
    new_filename = filename.replace(match.group(0), f"{{zpi$t={new_tags}}}")

    return new_filename

# メイン処理
def main():
    # logsフォルダー作成
    log_path = Path(LOG_DIR)
    log_path.mkdir(exist_ok=True)

    # ログファイル名（タイムスタンプ付き）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"判定結果_{timestamp}.txt"

    # NSFW/SFWフォルダーからファイルを収集
    nsfw_path = Path(NSFW_DIR)
    sfw_path = Path(SFW_DIR)

    nsfw_files = [(f, 'N') for f in nsfw_path.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    sfw_files = [(f, 'S') for f in sfw_path.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]

    all_files = nsfw_files + sfw_files

    print(f"\n処理対象: {len(all_files)}枚 (NSFW: {len(nsfw_files)}枚, SFW: {len(sfw_files)}枚)")
    print("=" * 80)

    # ログファイル初期化
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"WD14 Tagger 判定結果ログ\n")
        f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=" * 80 + "\n\n")

    # 判定結果を記録
    results = []

    for img_file, user_tag in all_files:
        try:
            # rating判定
            rating, ratings, nsfw_scores = get_rating(img_file)

            # SFW/NSFW判定
            wd14_tag, reason = determine_sfw_nsfw(ratings, nsfw_scores)

            # 一致判定
            match = "✓" if user_tag == wd14_tag else "✗"
            match_symbol = "🟢" if user_tag == wd14_tag else "🔴"

            # 結果表示（コンソール）
            print(f"{match_symbol} {img_file.name}")
            print(f"   あなた: {user_tag} | WD14: {wd14_tag} | {match}")
            print(f"   {reason}")
            print(f"   Rating: {rating} (explicit:{ratings.get('explicit', 0.0):.3f})")
            print()

            # ログファイルに詳細を書き込み
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{'='*80}\n")
                f.write(f"ファイル名: {img_file.name}\n")
                f.write(f"あなたの判定: {user_tag}\n")
                f.write(f"WD14の判定: {wd14_tag}\n")
                f.write(f"一致: {match}\n")
                f.write(f"判定理由: {reason}\n")
                f.write(f"\n--- Ratingスコア ---\n")
                for r_name, r_score in sorted(ratings.items()):
                    f.write(f"  {r_name}: {r_score:.4f}\n")
                f.write(f"\n--- NSFWタグスコア ---\n")
                for tag_name, tag_score in sorted(nsfw_scores.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {tag_name}: {tag_score:.4f}\n")
                f.write(f"\n")

            # 結果を記録
            results.append({
                'file': img_file.name,
                'user': user_tag,
                'wd14': wd14_tag,
                'match': user_tag == wd14_tag,
                'reason': reason
            })

        except Exception as e:
            error_msg = f"エラー: {img_file.name} - {e}"
            print(error_msg)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{error_msg}\n")
            continue

    # 統計表示
    print("=" * 80)
    print("判定結果サマリー:")
    print("=" * 80)

    total = len(results)
    matches = sum(1 for r in results if r['match'])
    accuracy = (matches / total * 100) if total > 0 else 0

    print(f"総ファイル数: {total}枚")
    print(f"一致: {matches}枚")
    print(f"不一致: {total - matches}枚")
    print(f"正解率: {accuracy:.1f}%")

    # 不一致リスト
    mismatches = [r for r in results if not r['match']]
    if mismatches:
        print(f"\n不一致リスト ({len(mismatches)}枚):")
        for r in mismatches:
            print(f"  - {r['file']}: あなた={r['user']}, WD14={r['wd14']}")

    # サマリーをログファイルに書き込み
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"判定結果サマリー\n")
        f.write(f"{'='*80}\n")
        f.write(f"総ファイル数: {total}枚\n")
        f.write(f"一致: {matches}枚\n")
        f.write(f"不一致: {total - matches}枚\n")
        f.write(f"正解率: {accuracy:.1f}%\n")

        if mismatches:
            f.write(f"\n不一致リスト ({len(mismatches)}枚):\n")
            for r in mismatches:
                f.write(f"  - {r['file']}: あなた={r['user']}, WD14={r['wd14']} ({r['reason']})\n")

    print(f"\nログファイル: {log_file}")

if __name__ == "__main__":
    main()