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

- **Python 3.11 - 3.13** （推奨: 3.12）
  - ⚠️ Python 3.14は現時点でonnxruntimeが未対応のため使用不可
  - Python 3.10以前は非推奨
- CUDA対応GPU（推奨、CPUでも動作可）
- 6GB+ VRAM（バッチサイズ4〜8推奨）

## インストール

### 1. Pythonバージョンの確認

```bash
# インストール済みのPythonバージョンを確認（Windows）
py --list

# インストール済みのPythonバージョンを確認（Linux/macOS）
python3 --version
```

Python 3.11〜3.13がインストールされていることを確認してください。

### 2. 仮想環境の作成

**Windows (PowerShell):**
```powershell
# プロジェクトディレクトリに移動
cd path\to\nsfw-gatekeeper

# Python 3.12で仮想環境を作成
py -3.12 -m venv wd14_env

# 仮想環境をアクティベート
.\wd14_env\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
# プロジェクトディレクトリに移動
cd path/to/nsfw-gatekeeper

# 仮想環境を作成
python3.12 -m venv wd14_env

# 仮想環境をアクティベート
source wd14_env/bin/activate
```

### 3. 依存パッケージのインストール

仮想環境がアクティブな状態で（プロンプトに `(wd14_env)` が表示されている状態）：

```bash
# pipを最新版にアップグレード
python -m pip install --upgrade pip

# 依存パッケージをインストール
python -m pip install -r requirements.txt
```

インストールには数分かかる場合があります。

## 起動方法

### 初回起動時

仮想環境をアクティベートしてから実行します。

**Windows (PowerShell):**
```powershell
# 仮想環境をアクティベート
.\wd14_env\Scripts\Activate.ps1

# プログラムを実行
python batch_tagger.py <入力ディレクトリ>
```

**Linux/macOS:**
```bash
# 仮想環境をアクティベート
source wd14_env/bin/activate

# プログラムを実行
python batch_tagger.py <入力ディレクトリ>
```

### 2回目以降

毎回、仮想環境のアクティベートが必要です。

**便利なTips（Windows）:**
個人用のバッチファイルを作成しておくと便利です：

```bat
@echo off
call .\wd14_env\Scripts\activate
python batch_tagger.py D:\your\image\folder -o D:\your\image\folder -b 8
pause
```

このファイルを `.gitignore` に追加すれば、個人設定がGitにコミットされません。

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

## トラブルシューティング

### ❌ ModuleNotFoundError: No module named 'cv2'

**原因:** 仮想環境がアクティベートされていないか、依存パッケージがインストールされていません。

**解決方法:**
```powershell
# 仮想環境をアクティベート
.\wd14_env\Scripts\Activate.ps1

# 依存パッケージを再インストール
python -m pip install -r requirements.txt
```

---

### ❌ ERROR: No matching distribution found for onnxruntime-gpu

**原因:** Python 3.14など、新しすぎるPythonバージョンを使用しています。

**解決方法:**
```powershell
# Python 3.12で仮想環境を作り直す
deactivate
Remove-Item -Recurse -Force .\wd14_env
py -3.12 -m venv wd14_env
.\wd14_env\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

---

### ❌ Fatal error in launcher: Unable to create process

**原因:** 古い仮想環境のパスが残っている、または仮想環境が破損しています。

**解決方法:**
```powershell
# 仮想環境を削除して作り直す
deactivate
Remove-Item -Recurse -Force .\wd14_env
py -3.12 -m venv wd14_env
.\wd14_env\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

---

### 💡 その他のヒント

- **プロンプトに `(wd14_env)` が表示されない場合:** 仮想環境がアクティベートされていません
- **処理が途中で止まる場合:** `--skip-processed` オプションで再開できます
- **GPU メモリ不足の場合:** `--batch-size 2` または `--device cpu` を試してください
- **ログを確認したい場合:** `logs/` フォルダ内のログファイルを参照

---

## バージョン

**v1.0.1** (2025-12-22)
- requirements.txt バージョン更新（Python 3.12対応）
- インストール・起動手順を詳細化
- トラブルシューティングセクション追加

**v1.0.0** (2025-12-22)
- 初回リリース
