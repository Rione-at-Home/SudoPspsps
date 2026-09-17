# SudoPsPsPs — 支援ロボット猫

SudoPsPsPs は、寄り添いと能動的な social interaction を提供するために設計された支援ロボット猫です。パン・チルト式ヘッド機構、コンピュータビジョン（Intel RealSense D455）、ローカル音声処理、そして Liquid Foundation Model（LFM）を用いて、ユーザーに視線を向け続けながら、感情を汲み取った対話を行います。

すべてのインタラクションは **音声コマンドと自然な会話** によって駆動されます。音声認識は Faster-Whisper によりローカルで処理し、負荷の大きい視覚処理と LLM 推論は専用の推論サーバーにオフロードしています。


![Alt Text](Images/Image%201.png)

![Alt Text](Images/Completed%20Build%20Image%201.png)

---

## ハードウェア

### パン・チルト機構

| 構成要素 | 説明 |
|-----------|-------------|
| パン用サーボ | Dynamixel XM430-W250-T（ID 1） |
| チルト用サーボ | Dynamixel XM430-W250-T（ID 2） |
| コントローラ | OpenCR |
| インターフェース | USB シリアル |
| プロトコル | Dynamixel Protocol 2.0 |

### センサ・音声系

| 構成要素 | 詳細 |
|-----------|---------|
| 視覚 | Intel RealSense D455 |
| マイク | Steinberg UR22mkII オーディオインターフェース |
| スピーカー | ローカルの ALSA 経由スピーカー |

---

## ソフトウェア構成

本システムは **ROS 2（Humble）** をミドルウェアとして、ハードウェア制御・知覚・推論をつないでいます。2 台のマシンが専用の Ethernet リンクで通信します。

| マシン | 役割 |
|---------|------|
| **Macbook**（推論サーバー） | `model_server.py` と LFM を実行。自然言語処理、タスクスケジューリングの推論、映像ベースの VAD 感情推定を担当。 |
| **Lenovo LOQ**（ロボットホスト） | すべての ROS 2 ノードを実行。ハードウェア制御（Dynamixel、RealSense）、STT/TTS、画像の前処理、行動ステートマシンを担当。 |
| **Ethernet `10.42.0.x`** | 画像クロップと JSON ペイロードを 2 台の間でやり取りする高速ローカルリンク。 |


![Alt Text](Images/rosgraph.png)
cat_brain_planner と lfm_bridge_node を除くすべてのノードは Lenovo LOQ 側で動作し、cat_brain_planner と lfm_bridge_node が Macbook 上の推論サーバーとのインターフェースとして機能します。

---

## コードベース概要

### ハードウェア制御

- **`dynamixel_driver.py`** — OpenCR との通信、トルクの有効化、起動時のキャリブレーションを担当。角度を Dynamixel 2.0 プロトコルの生の位置値に変換します。
- **`head_node.py`** — ROS 2 のハードウェア抽象化ノード。`/head/pan_target` と `/head/tilt_target` を購読し、モータ指令に指数移動平均（EMA）による平滑化を適用します。

### 視覚・追跡

- **`person_targetting_node.py`** — 視覚処理の中核となるステートマシン（`SEARCHING` → `TRACKING` → `TALKING`）。MediaPipe Pose によるバウンディングボックス検出に加え、離散的なサッカード、広めのデッドゾーン、好奇心を感じさせる首かしげといった猫らしい動きを実装しています。
- **`image_preprocessor.py`** — ユーザーをロックオンしている間のみ動作。RealSense のフレームをクロップ・リサイズし（人物クロップ 224×224、空間コンテキスト 336×336）、推論用にパブリッシュします。

### 音声・発話処理

- **`stt_node.py`** — GPU 上の `faster-whisper` によるローカル音声認識。RMS によるノイズ閾値ゲートを備え、認識結果を `/cat/stt_input` にパブリッシュします。これがロボットとのインタラクションの **主要なトリガー** です。
- **`tts_node.py`** — Piper によるオフライン音声合成。`/cat/robot_actions` を購読し、JSON ペイロードから `message_to_user` を抽出して、ローカルの ALSA システムへ音声を出力します。
- **`speaker_finder.py`** — PyAudio を用いたユーティリティ。オーディオデバイスを列挙し、正しいスピーカーのインデックスを特定します。

### AI・推論

- **`lfm_bridge_node.py`** — 非同期ブリッジノード。クロップしたユーザー画像を base64 ペイロードとしてまとめ、Macbook 上の視覚モデルへ POST します。Valence・Arousal・Dominance（VAD）スコアを `/cat/emotional_state` にパブリッシュします。
- **`cat_brain_planner.py`** — 認知の中枢。マルチターンの会話、予定の割り当て、能動的なリマインドを管理します。STT 入力と VAD の感情状態を統合し、気分に応じた提案（例：valence が低ければ休息を勧める）を生成します。
- **`train_lora.py`** — LFM2 視覚言語モデル向けの単体 QLoRA ファインチューニングスクリプト。HuggingFace の `peft` と `trl` を使用し、Sudo データセットでロボットのペルソナと推論を整合させます。

---

## セットアップと起動

### ネットワーク設定

両方のマシンを Ethernet で接続し、静的 IP を割り当てます:

| マシン | IP |
|---------|----|
| Macbook（推論サーバー） | `10.42.0.2` |
| Lenovo LOQ（ロボットホスト） | `10.42.0.1` |

---

### 1. Macbook — 推論サーバーの起動

```bash
python model_server.py --host 0.0.0.0 --port 8000
```

---

### 2. Lenovo LOQ — ワークスペースの source

```bash
source /opt/ros/humble/setup.bash
source ~/ri_one_master_ws/install/setup.bash
```

---

### 3. ハードウェアドライバの起動

```bash
ros2 launch realsense2_camera rs_launch.py
ros2 run sudo_pspsps head_node
```

---

### 4. 知覚・音声ノードの起動

それぞれ別のターミナルを開いて実行します:

```bash
ros2 run sudo_pspsps tts_node
```

---

### ヘッドの手動テスト

プランナーを起動する前にサーボの動作を確認する場合:

```bash
ros2 topic pub --once /head/pan_target std_msgs/msg/Float32 "{data: 30.0}"
```

---

## 現在のステータス

| 機能 | ステータス |
|---------|--------|
| Dynamixel 通信と ROS 2 ヘッドノード | 完了 |
| 猫らしいサッカード動作のプロファイリング | 完了 |
| MediaPipe による人物追跡とターゲティング | 完了 |
| ローカルの Whisper STT と Piper TTS | 完了 |
| Macbook への非同期 LFM HTTP ブリッジ | 完了 |
| RealSense のクロップ画像からの VAD 感情推定 | 完了 |
| 会話によるタスク計画と能動的な気分提案 | 完了 |
| LFM ペルソナ整合のための QLoRA 学習パイプライン | 完了 |

---

## チームのビジョン

SudoPsPsPs が目指すのは、注意を向け続け、会話の文脈を理解し、ユーザーに能動的に声をかけられるロボット・コンパニオンです。物理的なトリガーに反応するのではなく、音声入力・マルチモーダル知覚・高度な LLM 推論に導かれた、支えとなる社会的存在として振る舞うことを目標としています。