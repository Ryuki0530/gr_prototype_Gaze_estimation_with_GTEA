# Gaze Estimation with GTEA

このプロジェクトは、GTEAデータセットを用いた視線推定のプロトタイプです。
データセットは機械学習モデル作成用のものですが、このプロジェクトはこういったモデルを使わずに作成したアルゴリズムの検証の為のフレームワークとして作成しています。

## 概要

- GTEA＋ (Georgia Tech Egocentric Activity＋) データセットを利用
- 視線推定アルゴリズムの実装・評価
- Pythonによる実装

## 現状
- 現段階では、入力されたデータセットの映像に視線情報を重ねることしかできていません。

## ディレクトリ構成

```
.
├── data/           # データセット格納用
├── src/            # ソースコード
├── notebooks/      # 分析・実験用ノートブック
└── README.md
```

## 必要環境

- Python 3.8 以上
- 必要なライブラリは `requirements.txt` を参照

## セットアップ

```bash
git clone https://github.com/Ryuki0530/gr_prototype_Gaze_estimation_with_GTEA.git
cd gr_prototype_Gaze_estimation_with_GTEA
pip install -r requirements.txt
```

## 使い方

1. GTEAデータセットを `data/` ディレクトリに配置
2. メインスクリプトを実行

```bash
python src/main.py
```

## 各種アルゴリズムについて
-cnn_gru 
下記リポジトリを参照
https://github.com/Ryuki0530/gr_firstperson_gaze_estimetion_with_cnn_gru

## ライセンス

このプロジェクトはMITライセンスのもとで公開されています。

## 参考文献

- [GTEA Dataset](https://cbs.ic.gatech.edu/fpv/)


