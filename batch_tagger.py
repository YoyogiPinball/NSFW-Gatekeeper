"""
NSFW Gatekeeper - 大量画像の一括NSFW判定とフォルダ分け
GPU並列バッチ処理対応
"""

__version__ = "1.0.0"

import os
import csv
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from datetime import datetime

# win32com for creating .lnk shortcuts
try:
    import win32com.client
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

# 設定
IMAGE_SIZE = 448

# モデル設定
MODELS = {
    'wd-vit': {
        'repo': 'SmilingWolf/wd-vit-tagger-v3',
        'weight': 1.0
    },
    'joytag': {
        'repo': 'fancyfeast/joytag',
        'weight': 1.0
    }
}

# 判定があやしいの基準（ゆるめ）
UNCERTAIN_CRITERIA = {
    'explicit_range': (0.25, 0.45),  # explicitがこの範囲なら不確定
    'model_disagree': True,  # モデル間で判定が割れたら不確定
}


def create_shortcut(source_path, shortcut_path):
    """Windowsショートカット(.lnk)を作成"""
    if not HAS_WIN32:
        print(f"  スキップ: {source_path.name} (win32com未インストール)")
        return False

    try:
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(str(shortcut_path))
        shortcut.TargetPath = str(source_path.resolve())
        shortcut.WorkingDirectory = str(source_path.parent.resolve())
        shortcut.save()
        return True
    except Exception as e:
        print(f"  ショートカット作成エラー: {e}")
        return False


def handle_file(source_path, target_dir, mode, on_conflict='skip'):
    """ファイル操作（copy/move/shortcut）
    
    Returns:
        'success': 成功
        'skipped': スキップ（同名ファイル存在）
        'error': エラー
    """
    try:
        if mode == 'shortcut':
            shortcut_path = target_dir / f"{source_path.stem}.lnk"
            if shortcut_path.exists():
                if on_conflict == 'skip':
                    return 'skipped'
                else:  # rename
                    base = source_path.stem
                    counter = 1
                    while shortcut_path.exists():
                        shortcut_path = target_dir / f"{base}_{counter}.lnk"
                        counter += 1
            return 'success' if create_shortcut(source_path, shortcut_path) else 'error'
        elif mode == 'copy':
            dest_path = target_dir / source_path.name
            if dest_path.exists():
                if on_conflict == 'skip':
                    return 'skipped'
                else:  # rename
                    base = source_path.stem
                    ext = source_path.suffix
                    counter = 1
                    while dest_path.exists():
                        dest_path = target_dir / f"{base}_{counter}{ext}"
                        counter += 1
            shutil.copy2(source_path, dest_path)
            return 'success'
        elif mode == 'move':
            dest_path = target_dir / source_path.name
            if dest_path.exists():
                if on_conflict == 'skip':
                    return 'skipped'
                else:  # rename
                    base = source_path.stem
                    ext = source_path.suffix
                    counter = 1
                    while dest_path.exists():
                        dest_path = target_dir / f"{base}_{counter}{ext}"
                        counter += 1
            shutil.move(str(source_path), str(dest_path))
            return 'success'
        else:
            print(f"  不明なモード: {mode}")
            return 'error'
    except Exception as e:
        print(f"  ファイル操作エラー: {e}")
        return 'error'


def load_and_preprocess_wd(image_path):
    """WD用: 画像読み込みと前処理（並列実行用）"""
    try:
        image = Image.open(image_path).convert("RGB")
        return image_path, preprocess_image_wd(image)
    except Exception as e:
        return image_path, None


def load_and_preprocess_joytag(image_path):
    """JoyTag用: 画像読み込みと前処理（並列実行用）"""
    try:
        image = Image.open(image_path).convert("RGB")
        img = preprocess_image_joytag(image)
        return image_path, img[0]  # バッチ次元を除去
    except Exception as e:
        return image_path, None


def preprocess_image_wd(image):
    """WD14用の前処理"""
    image = np.array(image)
    image = image[:, :, ::-1]  # RGBからBGRへ

    # 正方形にパディング
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(
        image,
        ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)),
        mode="constant",
        constant_values=255
    )

    # リサイズ
    interp = cv2.INTER_AREA if size > IMAGE_SIZE else cv2.INTER_LANCZOS4
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)
    image = image.astype(np.float32)
    return image


def preprocess_image_joytag(image):
    """JoyTag用の前処理"""
    image = np.array(image)

    # 正方形にパディング
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(
        image,
        ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)),
        mode="constant",
        constant_values=255
    )

    # リサイズ
    interp = cv2.INTER_AREA if size > IMAGE_SIZE else cv2.INTER_LANCZOS4
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)
    image = image.astype(np.float32) / 255.0

    # CHW形式に変換
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)
    return image


class WDVitTagger:
    def __init__(self):
        print("WD-vit-tagger-v3をロード中...")
        model_path = hf_hub_download(MODELS['wd-vit']['repo'], "model.onnx")
        csv_path = hf_hub_download(MODELS['wd-vit']['repo'], "selected_tags.csv")

        self.session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        # タグ読み込み
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.rating_tags = {row['name']: i for i, row in enumerate(rows) if row['category'] == '9'}
        self.all_tags = [row['name'] for row in rows]

        # NSFWタグのインデックス
        self.nsfw_tag_indices = {}
        nsfw_names = ['nipples', 'pussy', 'penis', 'sex', 'cum', 'vaginal', 'oral', 'anal',
                      'nude', 'completely_nude', 'anus', 'testicles', 'spread_legs']
        for name in nsfw_names:
            if name in self.all_tags:
                self.nsfw_tag_indices[name] = self.all_tags.index(name)

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        img_array = preprocess_image_wd(image)
        img_array = np.expand_dims(img_array, 0)

        input_name = self.session.get_inputs()[0].name
        probs = self.session.run(None, {input_name: img_array})[0][0]

        # Rating取得
        ratings = {name: float(probs[idx]) for name, idx in self.rating_tags.items()}

        # NSFWタグ取得
        nsfw_scores = {name: float(probs[idx]) for name, idx in self.nsfw_tag_indices.items()}

        return ratings, nsfw_scores

    def predict_batch(self, image_paths):
        """バッチ推論"""
        # 並列で前処理
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(load_and_preprocess_wd, image_paths))

        # 有効な画像のみ抽出
        valid_paths = []
        valid_arrays = []
        for path, arr in results:
            if arr is not None:
                valid_paths.append(path)
                valid_arrays.append(arr)

        if not valid_arrays:
            return {}

        # バッチ推論
        batch_array = np.stack(valid_arrays, axis=0)
        input_name = self.session.get_inputs()[0].name
        all_probs = self.session.run(None, {input_name: batch_array})[0]

        # 結果をまとめる
        batch_results = {}
        for i, path in enumerate(valid_paths):
            probs = all_probs[i]
            ratings = {name: float(probs[idx]) for name, idx in self.rating_tags.items()}
            nsfw_scores = {name: float(probs[idx]) for name, idx in self.nsfw_tag_indices.items()}
            batch_results[path] = (ratings, nsfw_scores)

        return batch_results

    def judge(self, ratings, nsfw_scores):
        """NSFW判定"""
        explicit = ratings.get('explicit', 0)

        # パターン1: 性器タグ検出
        if nsfw_scores.get('penis', 0) >= 0.35 or nsfw_scores.get('pussy', 0) >= 0.35:
            return 'N', explicit

        # パターン2: 乳首検出
        if nsfw_scores.get('nipples', 0) >= 0.4:
            return 'N', explicit

        # パターン3: explicit高スコア
        if explicit >= 0.35:
            return 'N', explicit

        # パターン4: 複数NSFW要素
        high_nsfw = sum(1 for score in nsfw_scores.values() if score >= 0.3)
        if high_nsfw >= 2:
            return 'N', explicit

        return 'S', explicit


class JoyTagger:
    def __init__(self):
        print("JoyTagをロード中...")
        model_path = hf_hub_download(MODELS['joytag']['repo'], "model.onnx")
        label_path = hf_hub_download(MODELS['joytag']['repo'], "top_tags.txt")

        self.session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        with open(label_path, 'r', encoding='utf-8') as f:
            self.tags = [line.strip() for line in f.readlines()]

        # NSFWタグインデックス
        self.nsfw_tag_indices = {}
        nsfw_names = ['penis', 'pussy', 'vagina', 'nipples', 'nude', 'naked',
                      'sex', 'sexual', 'fucking', 'oral', 'fellatio', 'cunnilingus',
                      'paizuri', 'cum', 'nsfw', 'explicit', 'hentai']
        for name in nsfw_names:
            if name in self.tags:
                self.nsfw_tag_indices[name] = self.tags.index(name)

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        img_array = preprocess_image_joytag(image)

        input_name = self.session.get_inputs()[0].name
        probs = self.session.run(None, {input_name: img_array})[0][0]

        nsfw_scores = {name: float(probs[idx]) for name, idx in self.nsfw_tag_indices.items()}

        return {}, nsfw_scores

    def predict_batch(self, image_paths):
        """バッチ推論"""
        # 並列で前処理
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(load_and_preprocess_joytag, image_paths))

        # 有効な画像のみ抽出
        valid_paths = []
        valid_arrays = []
        for path, arr in results:
            if arr is not None:
                valid_paths.append(path)
                valid_arrays.append(arr)

        if not valid_arrays:
            return {}

        # バッチ推論
        batch_array = np.stack(valid_arrays, axis=0)
        input_name = self.session.get_inputs()[0].name
        all_probs = self.session.run(None, {input_name: batch_array})[0]

        # 結果をまとめる
        batch_results = {}
        for i, path in enumerate(valid_paths):
            probs = all_probs[i]
            nsfw_scores = {name: float(probs[idx]) for name, idx in self.nsfw_tag_indices.items()}
            batch_results[path] = ({}, nsfw_scores)

        return batch_results

    def judge(self, ratings, nsfw_scores):
        """NSFW判定"""
        # パターン1: 性器検出
        genital_tags = ['penis', 'pussy', 'vagina']
        for tag in genital_tags:
            if nsfw_scores.get(tag, 0) >= 0.3:
                return 'N', nsfw_scores.get('explicit', 0)

        # パターン2: 乳首検出
        if nsfw_scores.get('nipples', 0) >= 0.4:
            return 'N', nsfw_scores.get('explicit', 0)

        # パターン3: 性行為
        sex_tags = ['sex', 'sexual', 'fucking', 'fellatio', 'paizuri']
        for tag in sex_tags:
            if nsfw_scores.get(tag, 0) >= 0.4:
                return 'N', nsfw_scores.get('explicit', 0)

        # パターン4: explicit系
        if nsfw_scores.get('explicit', 0) >= 0.5 or nsfw_scores.get('nsfw', 0) >= 0.5:
            return 'N', nsfw_scores.get('explicit', 0)

        return 'S', nsfw_scores.get('explicit', 0)


def is_uncertain(judgements, explicit_scores):
    """不確定かどうか判定"""
    # モデル間で判定が割れている
    tags = [j[0] for j in judgements.values()]
    if len(set(tags)) > 1:
        return True, "モデル判定不一致"

    # explicitスコアが中途半端
    avg_explicit = sum(explicit_scores.values()) / len(explicit_scores) if explicit_scores else 0
    if UNCERTAIN_CRITERIA['explicit_range'][0] <= avg_explicit <= UNCERTAIN_CRITERIA['explicit_range'][1]:
        return True, f"explicit中途半端 ({avg_explicit:.2f})"

    return False, ""


def main():
    parser = argparse.ArgumentParser(
        description='大量画像のNSFW判定とフォルダ分け（GPU並列バッチ処理対応）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python batch_tagger.py D:\\images
  python batch_tagger.py D:\\images --output-dir E:\\sorted
  python batch_tagger.py D:\\images --batch-size 8
  python batch_tagger.py D:\\images --mode move
  python batch_tagger.py D:\\images -o D:\\images -b 4 -m copy --sfw-dir _SSSSS --nsfw-dir _NNNNN
        """
    )
    parser.add_argument('input_dir', help='処理対象ディレクトリ')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='出力ディレクトリ（デフォルト: ./origin）')
    parser.add_argument('--batch-size', '-b', type=int, default=4,
                        help='バッチサイズ（デフォルト: 4、6GB VRAMなら4推奨）')
    parser.add_argument('--mode', '-m', choices=['copy', 'move', 'shortcut'], default='copy',
                        help='ファイル操作モード（デフォルト: copy）')
    parser.add_argument('--sfw-dir', default='sfw',
                        help='SFW出力フォルダ名（デフォルト: sfw）')
    parser.add_argument('--nsfw-dir', default='nsfw',
                        help='NSFW出力フォルダ名（デフォルト: nsfw）')
    parser.add_argument('--unknown-dir', default='unknown',
                        help='不明出力フォルダ名（デフォルト: unknown）')
    parser.add_argument('--on-conflict', choices=['skip', 'rename'], default='skip',
                        help='同名ファイルがある場合の動作: skip(スキップ), rename(リネーム)（デフォルト: skip）')
    parser.add_argument('--limit', type=int, default=None,
                        help='処理件数制限（例: --limit 100）')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"エラー: ディレクトリが見つかりません: {input_dir}")
        return

    # 出力ベースディレクトリ
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        output_path = Path("origin")

    # 出力フォルダ作成
    sfw_dir = output_path / args.sfw_dir
    nsfw_dir = output_path / args.nsfw_dir
    unknown_dir = output_path / args.unknown_dir

    sfw_dir.mkdir(parents=True, exist_ok=True)
    nsfw_dir.mkdir(parents=True, exist_ok=True)
    unknown_dir.mkdir(parents=True, exist_ok=True)

    print(f"設定:")
    print(f"  入力: {input_dir}")
    print(f"  出力: {output_path}")
    print(f"    SFW: {sfw_dir}")
    print(f"    NSFW: {nsfw_dir}")
    print(f"    不明: {unknown_dir}")
    print(f"  バッチサイズ: {args.batch_size}")
    print(f"  モード: {args.mode}")
    print()

    # モデルロード
    taggers = {
        'wd-vit': WDVitTagger(),
        'joytag': JoyTagger()
    }

    # ファイル収集
    print(f"\n画像ファイルを収集中: {input_dir}")
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
        image_files.extend(input_dir.glob(f"**/*{ext}"))
        image_files.extend(input_dir.glob(f"**/*{ext.upper()}"))

    # 重複除去
    image_files = list(set(image_files))
    print(f"  {len(image_files)}枚の画像を発見")

    # 制限適用
    if args.limit:
        image_files = image_files[:args.limit]
        print(f"  制限適用: {args.limit}枚のみ処理")

    print("=" * 80)

    # 処理
    stats = {'sfw': 0, 'nsfw': 0, 'unknown': 0, 'skipped': 0, 'error': 0}
    batch_size = args.batch_size
    total_files = len(image_files)

    # バッチ単位で処理
    for batch_start in range(0, total_files, batch_size):
        batch_end = min(batch_start + batch_size, total_files)
        batch_files = image_files[batch_start:batch_end]

        try:
            # 各モデルでバッチ推論
            wd_results = taggers['wd-vit'].predict_batch(batch_files)
            joy_results = taggers['joytag'].predict_batch(batch_files)

            # 各ファイルについて判定
            for img_file in batch_files:
                try:
                    votes = {'N': 0, 'S': 0}
                    judgements = {}
                    explicit_scores = {}

                    # WD-ViT結果
                    if img_file in wd_results:
                        ratings, nsfw_scores = wd_results[img_file]
                        tag, explicit = taggers['wd-vit'].judge(ratings, nsfw_scores)
                        votes[tag] += MODELS['wd-vit']['weight']
                        judgements['wd-vit'] = (tag, explicit)
                        explicit_scores['wd-vit'] = explicit

                    # JoyTag結果
                    if img_file in joy_results:
                        ratings, nsfw_scores = joy_results[img_file]
                        tag, explicit = taggers['joytag'].judge(ratings, nsfw_scores)
                        votes[tag] += MODELS['joytag']['weight']
                        judgements['joytag'] = (tag, explicit)
                        explicit_scores['joytag'] = explicit

                    if not judgements:
                        stats['error'] += 1
                        continue

                    # 多数決
                    final_tag = 'N' if votes['N'] > votes['S'] else 'S'

                    # 不確定チェック
                    uncertain, reason = is_uncertain(judgements, explicit_scores)

                    if uncertain:
                        target_dir = unknown_dir
                        category = 'unknown'
                    elif final_tag == 'N':
                        target_dir = nsfw_dir
                        category = 'nsfw'
                    else:
                        target_dir = sfw_dir
                        category = 'sfw'

                    # ファイル操作
                    result = handle_file(img_file, target_dir, args.mode, args.on_conflict)
                    idx = batch_start + batch_files.index(img_file) + 1
                    if result == 'success':
                        stats[category] += 1
                        print(f"[{idx}/{total_files}] {category.upper()}: {img_file.name}")
                        if uncertain:
                            print(f"  理由: {reason}")
                    elif result == 'skipped':
                        stats['skipped'] += 1
                        # スキップは静かに処理（ログ出さない）
                    else:
                        stats['error'] += 1

                except Exception as e:
                    idx = batch_start + batch_files.index(img_file) + 1
                    print(f"[{idx}/{total_files}] エラー: {img_file.name} - {e}")
                    stats['error'] += 1

        except Exception as e:
            print(f"バッチ処理エラー ({batch_start+1}-{batch_end}): {e}")
            for img_file in batch_files:
                stats['error'] += 1

        # 進捗表示
        if batch_end % 50 == 0 or batch_end == total_files:
            print(f"\n進捗: {batch_end}/{total_files} 処理済み")
            print(f"  SFW: {stats['sfw']}, NSFW: {stats['nsfw']}, 不明: {stats['unknown']}, スキップ: {stats['skipped']}, エラー: {stats['error']}\n")

    # 最終サマリー
    print("\n" + "=" * 80)
    print("処理完了！")
    print("=" * 80)
    print(f"総処理: {total_files}枚")
    print(f"  SFW: {stats['sfw']}枚 → {sfw_dir}")
    print(f"  NSFW: {stats['nsfw']}枚 → {nsfw_dir}")
    print(f"  不明: {stats['unknown']}枚 → {unknown_dir}")
    print(f"  スキップ: {stats['skipped']}枚（同名ファイル）")
    print(f"  エラー: {stats['error']}枚")


if __name__ == "__main__":
    main()

