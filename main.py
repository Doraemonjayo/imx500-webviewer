from flask import Flask, Response
import cv2
import imx500_segmentation

# --- Flask MJPEG 配信用 --- #
app = Flask(__name__)

def gen_frames():
    while True:
        try:
            frame = imx500_segmentation.get_frame()
            ret, buffer = cv2.imencode('.jpg', frame[:, :, [2, 1, 0]])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception:
            pass

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- メイン処理 --- #
if __name__ == "__main__":
    imx500_segmentation.init()

    # Flask サーバーを起動
    app.run(host='0.0.0.0', port=5000)