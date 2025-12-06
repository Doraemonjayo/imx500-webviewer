# Portions of this software are based on:
# https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_object_detection_demo.py

from flask import Flask, Response
import cv2
import imx500_object_detection

# --- Flask MJPEG 配信用 --- #
app = Flask(__name__)

def gen_frames():
    while True:
        request = imx500_object_detection.picam2.capture_request()
        try:
            metadata = request.get_metadata()
            imx500_object_detection.last_results = imx500_object_detection.parse_detections(metadata)
            imx500_object_detection.draw_detections(request)
            frame = request.make_array("main")
            ret, buffer = cv2.imencode('.jpg', frame[:, :, [2, 1, 0]])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            request.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- メイン処理 --- #
if __name__ == "__main__":
    imx500_object_detection.init()

    # Flask サーバーを起動
    app.run(host='0.0.0.0', port=5000)