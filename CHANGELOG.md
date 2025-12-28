# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- README.md にバッジとgit cloneセクションを追加
- 改行コードをLFに統一（クロスプラットフォーム対応）
- .gitignore に開発用ファイル（.claude/等）を追加

### Released
- GitHub で公開リポジトリとして公開（2025-12-28）

## [1.0.1] - 2025-12-22

### Changed
- Python 3.12対応
- requirements.txt 全パッケージを最新版に更新
- ドキュメントの拡充

## [1.0.0] - 2025-12-22

### Added
- 初回リリース
- WD14タガー対応
- JoyTagタガー対応
- バッチ処理機能
- SFW/NSFW/Unknown自動分類
- 並列処理サポート
- ドライラン機能
- 詳細なログ出力

### Features
- 複数の画像認識モデル対応
- カスタマイズ可能なしきい値設定
- 柔軟な出力オプション（コピー/移動）
- CSV出力機能
