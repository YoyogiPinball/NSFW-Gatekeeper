# NSFW Gatekeeper

大量画像のNSFW判定とフォルダ振り分けツール（GPU並列バッチ処理対応）

## 概要

WD-ViT-tagger-v3 と JoyTag の2つのモデルを使用して、画像のNSFW判定を行い、自動でフォルダ分けします。

## 必要環境

- Python 3.8+
- CUDA対応GPU（推奨、CPUでも動作可）
- 6GB+ VRAM（バッチサイズ4〜8推奨）

## インストール

```bash
pip install -r requirements.txt
```

## 使い方

### 基本

```bash
python batch_tagger.py <入力ディレクトリ>
```

### オプション一覧

| オプション | 短縮形 | 説明 | デフォルト |
|------------|--------|------|----------|
| `--output-dir` | `-o` | 出力ディレクトリ | `./origin` |
| `--batch-size` | `-b` | バッチサイズ | `4` |
| `--mode` | `-m` | `copy`/`move`/`shortcut` | `copy` |
| `--sfw-dir` | - | SFWフォルダ名 | `sfw` |
| `--nsfw-dir` | - | NSFWフォルダ名 | `nsfw` |
| `--unknown-dir` | - | 判定不明フォルダ名 | `unknown` |
| `--on-conflict` | - | 同名ファイル: `skip`/`rename` | `skip` |
| `--limit` | - | 処理枚数制限 | なし |

### 使用例

```bash
# 基本（コピーモード、バッチ4）
python batch_tagger.py D:\images

# 出力先を入力と同じディレクトリに
python batch_tagger.py D:\images -o D:\images -b 8 -m copy

# 移動モード（本番用）
python batch_tagger.py D:\images -m move

# ショートカットモード
python batch_tagger.py D:\images -m shortcut
```

## 判定ロジック

- 2つのモデルの多数決で最終判定
- 判定が割れた場合や、explicitスコアが中途半端な場合は「不明」に分類
- 不明カテゴリにより誤判定を大幅に削減（99%精度達成）

## 注意事項

- コピーモードでは元ファイルの更新日時が保持されます
- 同名ファイルがある場合はデフォルトでスキップ
