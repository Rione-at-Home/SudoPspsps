
# import threading
# import numpy as np
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import String
# from faster_whisper import WhisperModel
# import pyaudio

# class SudoASRNode(Node):
#     def __init__(self):
#         super().__init__('sudo_asr_node')
#         self.publisher = self.create_publisher(String, '/cat/stt_input', 10)

#         # 1. Initialize Local Faster-Whisper
#         # device="cuda" uses your RTX 3050. compute_type="float16" is fast and accurate.
#         self.get_logger().info("Loading Faster-Whisper model...")
#         self.model = WhisperModel("small.en", device="cuda", compute_type="float16")
#         self.get_logger().info("Whisper model loaded ✓")

#         # Audio settings
#         self.CHUNK    = 1024
#         self.FORMAT   = pyaudio.paInt16 # Whisper prefers 16-bit PCM
#         self.CHANNELS = 1
#         self.RATE     = 16000

#         # Microphone setup
#         self.pa = pyaudio.PyAudio()
#         self.stream = self.pa.open(
#             format=self.FORMAT,
#             channels=self.CHANNELS,
#             rate=self.RATE,
#             input=True,
#             input_device_index=10, 
#             frames_per_buffer=self.CHUNK,
#         )

#         self.buffer      = []
#         self.buffer_lock = threading.Lock()

#         # Threads
#         self.capture_thread = threading.Thread(target=self._capture_audio, daemon=True)
#         self.capture_thread.start()
#         self.create_timer(3.0, self._process_and_publish)

#     def _capture_audio(self):
#         while rclpy.ok():
#             data = self.stream.read(self.CHUNK, exception_on_overflow=False)
#             with self.buffer_lock:
#                 self.buffer.append(data)

#     def _process_and_publish(self):
#         with self.buffer_lock:
#             if len(self.buffer) < 10: # Avoid processing tiny silences
#                 self.buffer = []
#                 return
#             audio_data  = b"".join(self.buffer)
#             self.buffer = []

#         # Convert raw bytes to float32 numpy array for Whisper
#         audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

#         # 2. Local Inference (No HTTP requests!)
#         segments, info = self.model.transcribe(audio_np, beam_size=5)
        
#         transcription = " ".join([segment.text for segment in segments])

#         if transcription.strip():
#             self.get_logger().info(f"Detected: '{transcription}'")
#             self.publisher.publish(String(data=transcription))

#     def destroy_node(self):
#         self.stream.stop_stream()
#         self.stream.close()
#         self.pa.terminate()
#         super().destroy_node()

# def main(args=None):
#     rclpy.init(args=args)
#     node = SudoASRNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


import threading
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from faster_whisper import WhisperModel
import pyaudio

class SudoASRNode(Node):
    def __init__(self):
        super().__init__('sudo_asr_node')
        self.publisher = self.create_publisher(String, '/cat/stt_input', 10)

        self.get_logger().info("Loading Faster-Whisper model...")
        self.model = WhisperModel("small.en", device="cuda", compute_type="float16")
        self.get_logger().info("Whisper model loaded ✓")

        # Audio settings
        self.CHUNK    = 1024
        self.FORMAT   = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE     = 16000

        # Noise threshold (0.0 - 1.0 scale, normalized float32)
        # 0.01 ≈ quiet room threshold. Raise to ~0.03 for noisier environments.
        self.NOISE_THRESHOLD = 0.08

        # Microphone setup
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            input_device_index=10,
            frames_per_buffer=self.CHUNK,
        )

        self.buffer      = []
        self.buffer_lock = threading.Lock()

        self.capture_thread = threading.Thread(target=self._capture_audio, daemon=True)
        self.capture_thread.start()
        self.create_timer(3.0, self._process_and_publish)

    def _capture_audio(self):
        while rclpy.ok():
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            with self.buffer_lock:
                self.buffer.append(data)

    def _compute_rms(self, audio_np: np.ndarray) -> float:
        """Compute Root Mean Square amplitude of the audio signal (0.0 - 1.0)."""
        return float(np.sqrt(np.mean(audio_np ** 2)))

    def _process_and_publish(self):
        with self.buffer_lock:
            if len(self.buffer) < 10:
                self.buffer = []
                return
            audio_data  = b"".join(self.buffer)
            self.buffer = []

        # Convert raw bytes → float32 numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # --- Noise threshold gate ---
        rms = self._compute_rms(audio_np)
        self.get_logger().debug(f"RMS amplitude: {rms:.4f} (threshold: {self.NOISE_THRESHOLD})")

        if rms < self.NOISE_THRESHOLD:
            self.get_logger().debug("Below noise threshold, skipping transcription.")
            return

        # Local inference
        segments, info = self.model.transcribe(audio_np, beam_size=5)
        transcription = " ".join([segment.text for segment in segments])

        if transcription.strip():
            self.get_logger().info(f"Detected: '{transcription}' (RMS: {rms:.4f})")
            self.publisher.publish(String(data=transcription))

    def destroy_node(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SudoASRNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()