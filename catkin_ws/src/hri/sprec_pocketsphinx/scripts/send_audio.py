#!/usr/bin/python

from time import sleep
import pyaudio
import rospy
from std_msgs.msg import *

def main():
    print("INITIALIZING AUDIO SENDER NODE BY MARCOSOFT...")
    rospy.init_node("audio_sender")
    pubSender = rospy.Publisher("/hri/sphinx_audio", UInt8MultiArray, queue_size=10)
    #print("AudioSender.->Waiting 5 seconds just because I can...")
    #sleep(5)
    stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    stream.start_stream()
    msg = UInt8MultiArray()
    while not rospy.is_shutdown():
        buf = stream.read(1024)
        if buf:
            msg.data = buf
            pubSender.publish(msg)


if __name__ == "__main__":
    main()
