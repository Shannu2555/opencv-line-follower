"""Subscribes to the camera, runs the OpenCV pipeline, publishes the line estimate
(/line/estimate: offset m, heading rad, curvature 1/m, look-ahead lateral m, confidence) and a debug image."""
import json
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from line_follower.camera_model import CameraModel
from line_follower.perception import LinePerception, DEFAULTS


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("line_perception")
        self.declare_parameter("calibration_file", "")
        for k, v in DEFAULTS.items():
            self.declare_parameter(k, v)
        params = {k: self.get_parameter(k).value for k in DEFAULTS}
        cam = CameraModel()  # mount pose (height, pitch) from the robot description
        K, D = cam.K, cam.D
        path = self.get_parameter("calibration_file").value
        if path:
            with open(path) as f:
                c = json.load(f)
            K, D = np.array(c["K"]), np.array(c["D"])
        self.pipe = LinePerception(K, D, cam.R, cam.C, (cam.w, cam.h), params)
        self.look_f = params["look_f"]
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Float32MultiArray, "line/estimate", 10)
        self.pub_dbg = self.create_publisher(Image, "line/debug_bev", 2)
        self.create_subscription(Image, "camera/image_raw", self.on_image, 5)

    def on_image(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        est = self.pipe.process(img, keep_debug=True)
        look = est.y_at(self.look_f) if est.coef is not None else 0.0
        out = Float32MultiArray()
        out.data = [float(est.ok), est.offset, est.heading, est.curvature, float(look), est.conf,
                    float(est.lost_frames), est.latency_ms]
        self.pub.publish(out)
        if self.pub_dbg.get_subscription_count():
            dbg = cv2.cvtColor(est.dbg["mask"] * 255, cv2.COLOR_GRAY2BGR)
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))


def main():
    rclpy.init()
    rclpy.spin(PerceptionNode())


if __name__ == "__main__":
    main()
