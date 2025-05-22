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
    print("INITIALIZING SPEECH RECOGNITION WITH POCKETSPHINX AND JSGF GRAMMAR BY MARCOSOFT...")
    rospy.init_node("sp_rec")
    pub_recognized = rospy.Publisher("/hri/sp_rec/recognized", RecognizedSpeech, queue_size=10)
    rospack = rospkg.RosPack()

    in_speech_bf = False
    hmm_folder = rospy.get_param("~hmm", "/usr/local/lib/python3.8/dist-packages/pocketsphinx/model/en-us")
    l_model    = rospy.get_param("~lm", "")
    dict_file  = rospy.get_param("~dict_file", "/usr/local/lib/python3.8/dist-packages/pocketsphinx/model/cmudict-en-us.dict")
    gram_file  = rospy.get_param("~gram_file", "")
    gram_rule  = rospy.get_param("~rule_name", "rule_name")
    gram_name  = rospy.get_param("~grammar_name", "grammar_name")

    print("SpRec.->HMM folder: " + hmm_folder)
    print("SpRec.->Dictionary file: " + dict_file)
    print("SpRec.->Grammar file: " + gram_file)
    print("SpRec.->Grammar name: " + gram_name)
    print("SpRec.->Rule name: " + gram_rule)
    print("SpRec.->Loading decoder with default config...")
    config = Decoder.default_config()
    config.set_string('-hmm', hmm_folder)
    config.set_string('-dict', dict_file)
    print("SpRec.->Initializing decoder using grammar: " + gram_file)
    decoder = Decoder(config)
    jsgf = Jsgf(gram_file)
    gram = gram_file[:len(gram_file)-5]
    print("SpRec.->Initializing jsgf grammar using rule: " + gram + '.' + gram_rule)
    rule = jsgf.get_rule(gram_name + '.' + gram_rule)
    fsg  = jsgf.build_fsg(rule, decoder.get_logmath(), 7.5)
    # print("SpRec.->Writing fsg to " + gram + '.fsg')
    # fsg.writefile(gram + '.fsg')
    decoder.set_fsg(gram, fsg)
    decoder.set_search(gram)
    decoder.start_utt()
    print("SpRec.->Decoder started successfully")
    rospy.Subscriber("/hri/sphinx_audio", UInt8MultiArray, callback_sphinx_audio)
    rospy.spin()

if __name__ == "__main__":
    main()
