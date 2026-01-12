from flask import Flask, Response
import cv2
import modlib_segmentation
import threading
import time

last_frame = None

# --- Flask MJPEG 配信用 --- #
app = Flask(__name__)

def gen_frames():
    global last_frame

    while True:
        if last_frame is None:
            time.sleep(0.005)
            continue
        
        try:
            frame = last_frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(e)
            time.sleep(0.005)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def onDetection(frame):
    global last_frame

    last_frame = frame

# --- メイン処理 --- #
if __name__ == "__main__":
    modlib_segmentation.init()
    ai_thread = threading.Thread(target=modlib_segmentation.run, args=(onDetection,))
    ai_thread.start()

    # Flask サーバーを起動
    app.run(host='0.0.0.0', port=5000)