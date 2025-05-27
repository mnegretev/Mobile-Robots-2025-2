 #!/usr/bin/env python3
import rospy
import speech_recognition as sr
from hri_msgs.msg import RecognizedSpeech

def main():
    rospy.init_node('speech_mic_node')
    pub = rospy.Publisher('/hri/sp_rec/recognized', RecognizedSpeech, queue_size=1)
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    rospy.loginfo("Micrófono listo. Habla una orden...")

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

    while not rospy.is_shutdown():
        with mic as source:
            rospy.loginfo("Escuchando...")
            audio = recognizer.listen(source)

        try:
            sentence = recognizer.recognize_google(audio).upper()
            rospy.loginfo("Reconocido: " + sentence)
            msg = RecognizedSpeech()
            msg.hypothesis = [sentence]
            pub.publish(msg)
        except sr.UnknownValueError:
            rospy.logwarn("No entendí lo que dijiste.")
        except sr.RequestError as e:
            rospy.logerr("Error al acceder al servicio de reconocimiento: {0}".format(e))

if __name__ == '__main__':
    main()
