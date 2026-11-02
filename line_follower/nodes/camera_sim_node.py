"""Publishes /camera/image_raw + /camera/camera_info from the 2D world and integrates /cmd_vel.
Also publishes ground truth on /ground_truth/cross_track_error so it never leaks into perception."""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from line_follower.camera_model import CameraModel
from line_follower.sim import World, Robot, Progress


class CameraSimNode(Node):
    def __init__(self):
        super().__init__("camera_sim")
        self.declare_parameter("track", "technical")
        self.declare_parameter("condition", "clean")
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("seed", 0)
        track = self.get_parameter("track").value
        cond = self.get_parameter("condition").value
        self.dt = 1.0 / self.get_parameter("rate_hz").value
        self.cam = CameraModel()
        self.world = World(track, cond, self.get_parameter("seed").value, self.cam)
        tr = self.world.track
        self.robot = Robot((tr.pts[0, 0], tr.pts[0, 1], tr.tangent[0]))
        self.prog = Progress(tr, 0)
        self.cmd = (0.0, 0.0)
        self.t = 0.0
        self.bridge = CvBridge()
        self.pub_img = self.create_publisher(Image, "camera/image_raw", 5)
        self.pub_info = self.create_publisher(CameraInfo, "camera/camera_info", 5)
        self.pub_cte = self.create_publisher(Float32, "ground_truth/cross_track_error", 10)
        self.create_subscription(Twist, "cmd_vel", self.on_cmd, 10)
        self.create_timer(self.dt, self.tick)

    def on_cmd(self, msg):
        self.cmd = (msg.linear.x, msg.angular.z)

    def tick(self):
        self.robot.step(*self.cmd, self.dt)
        self.t += self.dt
        img = self.world.camera_image(self.robot.pose, self.t)
        stamp = self.get_clock().now().to_msg()
        m = self.bridge.cv2_to_imgmsg(img, "bgr8")
        m.header.stamp = stamp
        m.header.frame_id = "camera_optical"
        info = CameraInfo()
        info.header = m.header
        info.width, info.height = self.cam.w, self.cam.h
        info.k = self.cam.K.ravel().tolist()
        info.d = self.cam.D.tolist()
        info.distortion_model = "plumb_bob"
        self.pub_img.publish(m)
        self.pub_info.publish(info)
        cte, _, _ = self.prog.update(*self.robot.pose)
        self.pub_cte.publish(Float32(data=float(cte)))


def main():
    rclpy.init()
    rclpy.spin(CameraSimNode())


if __name__ == "__main__":
    main()
