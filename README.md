# imx500-webviewer
RaspberryPiでのimx500の推論結果をwebから見るためのpythonスクリプトです。
パッケージ管理にはuvを使っています。

uvのインストール
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

# セットアップ
ドライバー関係のインストール(RaspberryPiのpicamera2とopencvはapt版)
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install imx500-all python3-picamera2 python3-opencv python3-openexr
sudo reboot
```

クローンしてセットアップ
```bash
git clone https://github.com/Doraemonjayo/imx500-webviewer.git
cd imx500-webviewer
uv venv --system-site-packages # picamera2とopencvをimportするため
uv sync
```

リポジトリが更新されたとき
```bash
# cd imx500-webviewer
git pull
uv sync
```

# 実行
実行には少し時間がかかります。
```bash
# cd imx500-webviewer
uv run main.py
```
実行後、`<RaspberryPiのIP>:5000/video_feed`を開いてください。

# クレジット
[このコード](https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_object_detection_demo.py)をベースにしました。

# ライセンス
[LICENSE](LICENSE)
