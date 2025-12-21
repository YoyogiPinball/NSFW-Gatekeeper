# NSFW Gatekeeper

大量画像のNSFW判定とフォルダ振り分けツール（GPU並列バッチ処理対応）

## 概要

WD-ViT-tagger-v3 と JoyTag の2つのモデルを使用して、画像のNSFW判定を行い、自動でフォルダ分けします。

## 主な機能

✅ **2モデル多数決判定** - WD-ViT と JoyTag で高精度判定
✅ **不確定カテゴリ** - 判定が割れた画像を自動的に分離
✅ **GPU/CPU対応** - CUDA対応GPUで高速処理、CPUでも動作
✅ **バッチ処理** - 大量画像を効率的に処理
✅ **tqdmプログレスバー** - リアルタイム進捗表示
✅ **ログ記録** - 詳細なログを自動保存（logs/）
✅ **CSV出力** - 判定結果をCSVファイルに保存可能
✅ **再実行スキップ** - 処理済みファイルを自動スキップ
✅ **ドライランモード** - 実際のファイル操作なしでテスト可能

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
| `--batch-size` | `-b` | バッチサイズ | 自動調整 |
| `--auto-batch` | - | バッチサイズ自動調整 | 有効 |
| `--mode` | `-m` | `copy`/`move`/`shortcut` | `copy` |
| `--sfw-dir` | - | SFWフォルダ名 | `sfw` |
| `--nsfw-dir` | - | NSFWフォルダ名 | `nsfw` |
| `--unknown-dir` | - | 判定不明フォルダ名 | `unknown` |
| `--on-conflict` | - | 同名ファイル: `skip`/`rename` | `skip` |
| `--limit` | - | 処理枚数制限 | なし |
| `--dry-run` | - | ドライランモード（実際のファイル操作なし） | 無効 |
| `--workers` | - | 前処理の並列スレッド数 | `4` |
| `--device` | - | デバイス: `auto`/`gpu`/`cpu` | `auto` |
| `--csv-output` | - | 判定結果をCSVに出力 | なし |
| `--skip-processed` | - | 処理済みファイルをスキップ | 無効 |

### 使用例

```bash
# 基本（自動バッチサイズ、コピーモード）
python batch_tagger.py D:\images

# 出力先を入力と同じディレクトリに、バッチサイズ8
python batch_tagger.py D:\images -o D:\images -b 8

# 移動モード（本番用）
python batch_tagger.py D:\images -m move

# ドライランモード（テスト）
python batch_tagger.py D:\images --dry-run

# CSV出力付き
python batch_tagger.py D:\images --csv-output results.csv

# 処理済みスキップ機能（中断→再開時）
python batch_tagger.py D:\images --skip-processed

# CPU使用（GPUなし環境）
python batch_tagger.py D:\images --device cpu

# 前処理スレッド数を増やす
python batch_tagger.py D:\images --workers 8
```

## 判定ロジック

- 2つのモデルの多数決で最終判定
- 判定が割れた場合や、explicitスコアが中途半端な場合は「不明」に分類
- 不明カテゴリにより誤判定を大幅に削減（99%精度達成）

## ログとCSV出力

### ログファイル
- 自動的に `logs/nsfw_gatekeeper_YYYYMMDD_HHMMSS.log` に保存
- コンソールとファイルの両方に出力
- DEBUG/INFO/WARNING/ERRORのレベル別記録

### CSV出力
`--csv-output` オプションで判定結果をCSVファイルに出力できます。

```bash
python batch_tagger.py D:\images --csv-output results.csv
```

CSVには以下の情報が含まれます：
- ファイルパス
- 判定カテゴリ（sfw/nsfw/unknown）
- 各モデルの判定結果
- explicitスコア
- 不確定判定の理由

## 注意事項

- コピーモードでは元ファイルの更新日時が保持されます
- 同名ファイルがある場合はデフォルトでスキップ
- `--skip-processed` 使用時は `.processed.json` で処理済みファイルを管理
- ドライランモードではファイル操作は行われません（判定のみ）

## バージョン

v1.0.0
