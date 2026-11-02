"""Turns /line/estimate into /cmd_vel with the chosen steering law. Stops if the line is lost too long."""
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from line_follower.controllers import CONTROLLERS
from line_follower.perception import LineEstimate


class ControlNode(Node):
    def __init__(self):
        super().__init__("line_control")
        self.declare_parameter("controller", "stanley_ff")
        self.declare_parameter("speed", 0.19)
        self.declare_parameter("max_lost_frames", 30)
        name = self.get_parameter("controller").value
        if name == "centroid_pid":
            raise ValueError("centroid_pid needs the centroid detector, use the experiment script for it")
        self.ctrl = CONTROLLERS[name](self.get_parameter("speed").value)
        self.max_lost = self.get_parameter("max_lost_frames").value
        self.pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(Float32MultiArray, "line/estimate", self.on_est, 10)

    def on_est(self, msg):
        ok, off, head, curv, look, conf, lost, _ = msg.data
        est = LineEstimate()
        est.ok = bool(ok)
        est.offset, est.heading, est.curvature = off, head, curv
        # rebuild a quadratic that matches offset, heading and curvature at f = 0

        c1 = math.tan(head)
        est.coef = (off, c1, curv * (1 + c1 * c1) ** 1.5 / 2)
        est.lost_frames = int(lost)
        v, w = self.ctrl(est)
        cmd = Twist()
        if est.lost_frames <= self.max_lost:
            cmd.linear.x, cmd.angular.z = float(v), float(w)
        self.pub.publish(cmd)


def main():
    rclpy.init()
    rclpy.spin(ControlNode())


if __name__ == "__main__":
    main()
