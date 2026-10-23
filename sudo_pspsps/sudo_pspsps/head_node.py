#!/usr/bin/env python3
"""Head movement and read-only motor health reporting for ROS 2 Humble.

Publishes /head_health_status (diagnostic_msgs/DiagnosticArray) at 1 Hz.
The head Dynamixel bus MUST be distinct from the Kobuki's serial port.
"""

import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

from .dynamixel_driver import (
    DynamixelDriver, DynamixelError,
    ADDR_HARDWARE_ERROR, ADDR_TORQUE_ENABLE,
    ADDR_PRESENT_TEMPERATURE,
)


class HeadNode(Node):
    def __init__(self):
        super().__init__('head_node')

        self.declare_parameter('head_device', '/dev/ttyUSB1')
        self.declare_parameter('health_rate_hz', 1.0)
        # Provisional values, NOT certified XM430 safety limits.
        self.declare_parameter('temperature_warn_c', 65.0)
        self.declare_parameter('temperature_error_c', 75.0)
        device = self.get_parameter('head_device').value
        rate = float(self.get_parameter('health_rate_hz').value)
        self.warn_c = float(self.get_parameter('temperature_warn_c').value)
        self.error_c = float(self.get_parameter('temperature_error_c').value)
        if not 0 < self.warn_c < self.error_c:
            raise ValueError('Temperature thresholds must satisfy 0 < warn < error')

        self.health_pub = self.create_publisher(
            DiagnosticArray, '/head_health_status', 10
        )
        self.driver = None
        self.movement_error = None
        self.started = time.monotonic()
        self.current_pan = self.current_tilt = 0.0
        self.target_pan = self.target_tilt = 0.0

        # Keep control/health feedback on the same serial-port owner.
        # Startup failures should fail visibly, rather than announce HEALTHY.
        self.driver = DynamixelDriver(device_name=device)
        try:
            self.driver.calibrate_zero()
            self.driver.enable()
        except Exception:
            self.driver.close()
            self.driver = None
            raise

        self.create_subscription(Float32, '/head/pan_target', self.pan_callback, 10)
        self.create_subscription(Float32, '/head/tilt_target', self.tilt_callback, 10)
        self.create_timer(0.05, self.control_loop)
        self.create_timer(1.0 / max(rate, 0.1), self.publish_health)
        self.get_logger().info(f'Head control + diagnostics started on {device}')
        
    def pan_callback(self, msg):
        if math.isfinite(msg.data):
            self.target_pan = float(msg.data)
        else:
            self.get_logger().warning('Ignoring non-finite pan target')

    def tilt_callback(self, msg):
        if math.isfinite(msg.data):
            self.target_tilt = float(msg.data)
        else:
            self.get_logger().warning('Ignoring non-finite tilt target')

    def control_loop(self):
        if self.driver is None or self.movement_error is not None:
            return  # Do not keep issuing motion writes after a bus failure.
        alpha = 0.15
        self.current_pan += (self.target_pan - self.current_pan) * alpha
        self.current_tilt += (self.target_tilt - self.current_tilt) * alpha
        try:
            self.driver.set_pan(self.current_pan)
            self.driver.set_tilt(self.current_tilt)
        except (DynamixelError, OSError) as exc:
            self.movement_error = str(exc)
            self.get_logger().error(f'Movement stopped: {exc}')
            self.publish_health()  # Issue a fault immediately.

    @staticmethod
    def status(name, level, message, values=None):
        status = DiagnosticStatus()
        status.name = name
        status.hardware_id = 'head_xm430'
        status.level = level
        status.message = message
        status.values = [KeyValue(key=str(k), value=str(v)) for k, v in (values or {}).items()]
        return status

    def motor_health(self, name, dxl_id):
        """Read status individually; one failing motor should not hide the other."""
        if self.driver is None:
            return self.status(f'head/{name}', DiagnosticStatus.STALE,
                               'Driver unavailable')
        try:
            # The driver checks Dynamixel communication and response errors.
            hw = self.driver._read1(dxl_id, ADDR_HARDWARE_ERROR, 'hardware error')
            temp = self.driver._read1(dxl_id, ADDR_PRESENT_TEMPERATURE, 'temperature')
            torque = self.driver._read1(dxl_id, ADDR_TORQUE_ENABLE, 'torque enable')
            position = self.driver.get_pan() if name == 'pan' else self.driver.get_tilt()
            values = {'motor_id': dxl_id, 'temperature_c': temp,
                      'torque_enabled': torque, 'hardware_error_bits': hex(hw),
                      'position_degrees': f'{position:.2f}'}
            if hw:
                level, msg = DiagnosticStatus.ERROR, f'Hardware error bits set: 0x{hw:02X}'
            elif temp >= self.error_c:
                level, msg = DiagnosticStatus.ERROR, f'Temperature above configured error threshold ({self.error_c:g} C)'
            elif temp >= self.warn_c:
                level, msg = DiagnosticStatus.WARN, f'Temperature above configured warning threshold ({self.warn_c:g} C)'
            elif torque != 1:
                level, msg = DiagnosticStatus.WARN, 'Torque disabled unexpectedly'
            else:
                level, msg = DiagnosticStatus.OK, 'Motor responding; feedback nominal'
            return self.status(f'head/{name}', level, msg, values)
        except (DynamixelError, OSError) as exc:
            return self.status(f'head/{name}', DiagnosticStatus.ERROR,
                               f'Motor feedback read failed: {exc}', {'motor_id': dxl_id})

    def publish_health(self):
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        if self.driver is None:
            msg.status = [self.status('head/driver', DiagnosticStatus.STALE,
                                      'Head motor driver unavailable')]
        else:
            # Important: polling motor registers uses the same bus as motion.
            msg.status = [
                self.motor_health('pan', self.driver.pan_id),
                self.motor_health('tilt', self.driver.tilt_id),
            ]
            if self.movement_error is not None:
                msg.status.append(self.status('head/movement', DiagnosticStatus.ERROR,
                                              self.movement_error))
        self.health_pub.publish(msg)

    def destroy_node(self):
        try:
            if self.driver is not None:
                try:
                    self.driver.disable()
                except Exception as exc:
                    self.get_logger().warning(f'Could not disable torque: {exc}')
                self.driver.close()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = HeadNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

