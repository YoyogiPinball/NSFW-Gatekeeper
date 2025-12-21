import os
import re
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from datetime import datetime

# 設定
NSFW_DIR = "_moto/nsfw"  # NSFW画像フォルダ（手動振り分け）
SFW_DIR = "_moto/sfw"    # SFW画像フォルダ（手動振り分け）
LOG_DIR = "logs"  # ログ出力フォルダ
MODEL_REPO = "fancyfeast/joytag"  # JoyTagモデル

# JoyTagモデルのロード
print("JoyTagモデルをダウンロード中...")
model_path = hf_hub_download(MODEL_REPO, "model.onnx")
label_path = hf_hub_download(MODEL_REPO, "top_tags.txt")

print("モデルをロード中...")
session = ort.InferenceSession(
    model_path,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# ラベル読み込み
with open(label_path, 'r', encoding='utf-8') as f:
    tags = [line.strip() for line in f.readlines()]

print(f"総タグ数: {len(tags)}")

# NSFWタグのインデックスを探す
nsfw_tag_indices = {}
nsfw_tag_names = [
    # 性器・露出
    'penis', 'pussy', 'vagina', 'nipples', 'breasts', 'nude', 'naked',
    'areola', 'areolae', 'genitals', 'clitoris', 'anus',
    # 性行為
    'sex', 'sexual', 'intercourse', 'penetration', 'fucking',
    'oral', 'fellatio', 'blowjob', 'cunnilingus',
    'paizuri', 'titfuck', 'boobjob',
    'vaginal', 'anal sex',
    # その他
    'cum', 'semen', 'ejaculation', 'orgasm',
    'masturbation', 'erection', 'spread legs', 'spread pussy',
    'nsfw', 'explicit', 'hentai', 'pornography'
]

for tag_name in nsfw_tag_names:
    try:
        idx = tags.index(tag_name)
        nsfw_tag_indices[tag_name] = idx
    except ValueError:
        pass

print(f"検出可能なNSFWタグ: {len(nsfw_tag_indices)}個")
print(f"  {list(nsfw_tag_indices.keys())[:10]}...")  # 最初の10個を表示

def preprocess_image(image_path):
    """画像を前処理"""
    img = Image.open(image_path).convert('RGB')
    # JoyTagの入力サイズは448x448
    img = img.resize((448, 448), Image.LANCZOS)
    img_array = np.array(img).astype(np.float32) / 255.0
    # CHW形式に変換
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, 0)  # バッチ次元追加
    return img_array

def get_tags(image_path, threshold=0.3):
    """画像からタグを取得"""
    img_array = preprocess_image(image_path)

    # 推論
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_array})[0][0]

    # NSFWタグスコア取得
    nsfw_scores = {}
    for tag_name, idx in nsfw_tag_indices.items():
        nsfw_scores[tag_name] = float(output[idx])

    # 閾値以上のタグを取得
    detected_tags = [(tags[i], float(output[i])) for i in range(len(output)) if output[i] >= threshold]
    detected_tags.sort(key=lambda x: x[1], reverse=True)

    return nsfw_scores, detected_tags

def determine_sfw_nsfw(nsfw_scores):
    """SFW/NSFW判定"""

    # 優先度1: 性器が明確に見える
    genital_tags = ['penis', 'pussy', 'vagina', 'clitoris', 'genitals']
    for tag in genital_tags:
        if nsfw_scores.get(tag, 0) >= 0.3:
            return 'N', f"性器検出: {tag}:{nsfw_scores[tag]:.3f}"

    # 優先度2: 乳首が見える
    nipple_tags = ['nipples', 'areola', 'areolae']
    for tag in nipple_tags:
        if nsfw_scores.get(tag, 0) >= 0.4:
            return 'N', f"乳首検出: {tag}:{nsfw_scores[tag]:.3f}"

    # 優先度3: 性行為
    sex_tags = ['sex', 'sexual', 'intercourse', 'penetration', 'fucking',
                'fellatio', 'blowjob', 'cunnilingus', 'paizuri', 'titfuck']
    for tag in sex_tags:
        if nsfw_scores.get(tag, 0) >= 0.4:
            return 'N', f"性行為検出: {tag}:{nsfw_scores[tag]:.3f}"

    # 優先度4: explicit系タグ
    explicit_tags = ['nsfw', 'explicit', 'hentai', 'pornography']
    for tag in explicit_tags:
        if nsfw_scores.get(tag, 0) >= 0.5:
            return 'N', f"Explicit検出: {tag}:{nsfw_scores[tag]:.3f}"

    # 優先度5: 全裸 + 性的要素
    nude_score = max(nsfw_scores.get('nude', 0), nsfw_scores.get('naked', 0))
    sexual_element = any(nsfw_scores.get(tag, 0) >= 0.2 for tag in ['nipples', 'pussy', 'penis'])
    if nude_score >= 0.6 and sexual_element:
        return 'N', f"全裸+性的要素検出: nude:{nude_score:.3f}"

    # 優先度6: 複数のNSFWタグが中程度のスコア
    medium_nsfw_count = sum(1 for score in nsfw_scores.values() if score >= 0.25)
    if medium_nsfw_count >= 3:
        return 'N', f"複数NSFW要素検出: {medium_nsfw_count}個のタグ >= 0.25"

    # それ以外はSFW
    return 'S', "NSFW要素検出されず"

def main():
    # logsフォルダー作成
    log_path = Path(LOG_DIR)
    log_path.mkdir(exist_ok=True)

    # ログファイル名（タイムスタンプ付き）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"JoyTag判定結果_{timestamp}.txt"

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
        f.write(f"JoyTag 判定結果ログ\n")
        f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=" * 80 + "\n\n")

    # 判定結果を記録
    results = []

    for img_file, user_tag in all_files:
        try:
            # タグ取得
            nsfw_scores, detected_tags = get_tags(img_file)

            # SFW/NSFW判定
            joytag_tag, reason = determine_sfw_nsfw(nsfw_scores)

            # 一致判定
            match = "✓" if user_tag == joytag_tag else "✗"
            match_symbol = "🟢" if user_tag == joytag_tag else "🔴"

            # 結果表示（コンソール）
            print(f"{match_symbol} {img_file.name}")
            print(f"   あなた: {user_tag} | JoyTag: {joytag_tag} | {match}")
            print(f"   {reason}")
            print()

            # ログファイルに詳細を書き込み
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{'='*80}\n")
                f.write(f"ファイル名: {img_file.name}\n")
                f.write(f"あなたの判定: {user_tag}\n")
                f.write(f"JoyTagの判定: {joytag_tag}\n")
                f.write(f"一致: {match}\n")
                f.write(f"判定理由: {reason}\n")
                f.write(f"\n--- NSFWタグスコア（上位15個）---\n")
                for tag_name, tag_score in sorted(nsfw_scores.items(), key=lambda x: x[1], reverse=True)[:15]:
                    if tag_score > 0.01:
                        f.write(f"  {tag_name}: {tag_score:.4f}\n")
                f.write(f"\n--- 検出タグ（上位10個）---\n")
                for tag, score in detected_tags[:10]:
                    f.write(f"  {tag}: {score:.4f}\n")
                f.write(f"\n")

            # 結果を記録
            results.append({
                'file': img_file.name,
                'user': user_tag,
                'joytag': joytag_tag,
                'match': user_tag == joytag_tag,
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
            print(f"  - {r['file']}: あなた={r['user']}, JoyTag={r['joytag']}")

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
                f.write(f"  - {r['file']}: あなた={r['user']}, JoyTag={r['joytag']} ({r['reason']})\n")

    print(f"\nログファイル: {log_file}")

if __name__ == "__main__":
    main()
