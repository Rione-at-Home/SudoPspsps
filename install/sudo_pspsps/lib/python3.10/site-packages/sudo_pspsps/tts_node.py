#
## tts_node.py   ── ACT (speech)
#
#  This is the third step of the see-think-act cycle — for speech.
#
#  What this node does:
#    1. Listens for action messages from the brain node.
#    2. Extracts the text to speak from the message.
#    3. Uses Piper (a local, offline text-to-speech engine) to speak it
#       through the computer's speakers.
#
#  Piper runs entirely on this machine — no internet connection needed.
#  You point it at a .onnx model file and it synthesises speech in real time.
#
#  Subscribed topics:
#    /cat/robot_actions  — String (JSON with a "message_to_user" key)
#
#  Message format expected on /cat/robot_actions:
#    {"message_to_user": "Hello! Nice to meet you."}
#

from matplotlib import text

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from piper import PiperVoice
import io
import wave
import pyaudio
import json


class TTSNode(Node):

    # Path to the Piper voice model file.
    # Download voices from: https://github.com/rhasspy/piper/releases
    MODEL_PATH = "/home/ri-one/SudoPsPsPs/SudoPspsps/sudo_pspsps/sudo_pspsps/en_US-amy-low.onnx"

    def __init__(self):
        super().__init__('tts_node')

        # ── Load the Piper voice model ────────────────────────────────────────
        self.get_logger().info("Loading Piper voice model...")
        self.voice = PiperVoice.load(self.MODEL_PATH)
        self.get_logger().info("Voice model ready.")

        # ── Set up the audio output stream ────────────────────────────────────
        # PyAudio sends raw PCM audio to your speakers.
        # Piper outputs 16-bit, mono audio at 22050 Hz.
        self.pa = pyaudio.PyAudio()
        self.audio_stream = self.pa.open(
            format   = pyaudio.paInt16,
            channels = 1,
            rate     = 22050,
            output   = True,
        )

        # ── Subscriber ────────────────────────────────────────────────────────
        self.create_subscription(
            String,
            '/cat/robot_actions',
            self.actions_callback,
            10,
        )

        self.get_logger().info("TTS node ready — listening for messages to speak.")


    def actions_callback(self, msg):
        """
        Called when a message arrives on /cat/robot_actions.

        The brain node publishes JSON like:
            {"message_to_user": "Hello! Nice to meet you."}

        We parse that JSON, extract the text, and speak it.
        """
        try:
            data = json.loads(msg.data)
            text = data.get("message_to_user", "").strip()

            if text:
                self.get_logger().info(f"Speaking: \"{text}\"")
                self._speak(text)

        except json.JSONDecodeError as e:
            self.get_logger().error(f"Could not parse message as JSON: {e}")


    def _speak(self, text):
        """
        Convert `text` to speech using Piper and play it through the speakers.
        """
        import tempfile
        import os

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav_path = temp_wav.name

            #Open it cleanly via the wave module so Piper receives a valid WAV file structure
            with wave.open(temp_wav_path, 'wb') as wav_file:
                wav_file.setnchannels(1)      # Mono
                wav_file.setsampwidth(2)     # 16-bit audio = 2 bytes per sample
                wav_file.setframerate(22050) # Piper default sample rate
                
                # Piper writes the generated speech into this real file object
                self.voice.synthesize(text, wav_file)

            with wave.open(temp_wav_path, 'rb') as wav_file:
                pcm_data = wav_file.readframes(wav_file.getnframes())

            if pcm_data:
                self.audio_stream.write(pcm_data)
                
            else:
                self.get_logger().warn("Audio generation produced an empty frame sequence.")

        except Exception as e:
            self.get_logger().error(f"Failed to synthesize speech: {e}")

        finally:
            # 5. Cleanup: Always ensure the temporary file is removed from disk
            if 'temp_wav_path' in locals() and os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
        

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        """Stop the audio stream cleanly when the node shuts down."""
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.pa.terminate()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()