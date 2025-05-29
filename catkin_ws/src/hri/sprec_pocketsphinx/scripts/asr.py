#!/usr/bin/env python

import os
import rospy
import rospkg
from std_msgs.msg import UInt8MultiArray
from hri_msgs.msg import RecognizedSpeech
from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *

def callback_sphinx_audio(msg):
    global decoder, in_speech_bf, pub_recognized
    decoder.process_raw(msg.data, False, False)
    if decoder.get_in_speech() != in_speech_bf:
        in_speech_bf = decoder.get_in_speech()
        if not in_speech_bf:
            decoder.end_utt()
            if decoder.hyp() != None:
                hyp = decoder.hyp()
                print("SpRec.->Recognized: " + "'" + hyp.hypstr + "' with p=" + str(hyp.prob))
                recog_sp = RecognizedSpeech()
                recog_sp.hypothesis.append(hyp.hypstr)
                recog_sp.confidences.append(hyp.prob)
                pub_recognized.publish(recog_sp)
            decoder.start_utt()

def main():
    global decoder, in_speech_bf, pub_recognized
    print("INITIALIZING SPEECH RECOGNITION WITH POCKETSPHINX...")

    rospy.init_node("sp_rec")
    pub_recognized = rospy.Publisher("/hri/sp_rec/recognized", RecognizedSpeech, queue_size=10)

    hmm_folder = rospy.get_param("~hmm", "/usr/share/pocketsphinx/model/en-us/en-us")
    dict_file  = rospy.get_param("~dict_file", "")
    gram_file  = rospy.get_param("~gram_file", "")
    gram_rule  = rospy.get_param("~rule_name", "move2")
    gram_name  = rospy.get_param("~grammar_name", "voice_cmd")

    if not os.path.exists(gram_file):
        rospy.logerr("Grammar file not found: %s", gram_file)
        return
    if not os.path.exists(dict_file):
        rospy.logerr("Dictionary file not found: %s", dict_file)
        return

    print(f"HMM: {hmm_folder}")
    print(f"DICT: {dict_file}")
    print(f"GRAM: {gram_file} | RULE: {gram_rule}")

    config = Decoder.default_config()
    config.set_string('-hmm', hmm_folder)
    config.set_string('-dict', dict_file)

    decoder = Decoder(config)
    jsgf = Jsgf(gram_file)
    rule = jsgf.get_rule(f"{gram_name}.{gram_rule}")
    fsg = jsgf.build_fsg(rule, decoder.get_logmath(), 7.5)
    gram_base = os.path.splitext(os.path.basename(gram_file))[0]
    decoder.set_fsg(gram_base, fsg)
    decoder.set_search(gram_base)
    decoder.start_utt()

    in_speech_bf = False
    rospy.Subscriber("/hri/sphinx_audio", UInt8MultiArray, callback_sphinx_audio)
    print("Speech recognition node ready.")
    rospy.spin()
if __name__ == "__main__":
    main()
