#!/usr/bin/env python

import rospy
from cse_190_assi_1.msg import RobotProbabilities
from std_msgs.msg import String, Float32, Bool
import json


class RobotLogger():
    def __init__(self):
        rospy.init_node("robot_logger")
        self.temp_results_sub = rospy.Subscriber(
                "/results/temperature_data",
                Float32,
                self.handle_incoming_temperature_data
        )
        self.tex_results_sub = rospy.Subscriber(
                "/results/texture_data",
                String,
                self.handle_incoming_texture_data
        )
        self.prob_results_sub = rospy.Subscriber(
                "/results/probabilities",
                RobotProbabilities,
                self.handle_incoming_probabilities
        )
        self.simulation_complete_sub = rospy.Subscriber(
                "/map_node/sim_complete",
                Bool,
                self.handle_shutdown
        )
        self.init_files()
        rospy.spin()

    def init_files(self):
        with open('texture_results.json', 'w+') as infile:
            pass
        open('temperature_results.json', 'w+').close()
        open('probability_results.json', 'w+').close()
        self.texture_data = []
        self.temperature_data = []
        self.probability_data = []

    def handle_incoming_texture_data(self, message):
        texture = message.data
        self.texture_data.append(texture)

    def handle_incoming_temperature_data(self, message):
        temperature = message.data
        self.temperature_data.append(temperature)

    def handle_incoming_probabilities(self, message):
        probabilities_list = [round(x, 5) for x in message.data]
        self.probability_data.append(probabilities_list)

    def handle_shutdown(self, message):
        print "sim complete!", message.data
        if message.data:
            with open('texture_results.json', 'w') as tex:
                json.dump(self.texture_data, tex)
            with open('temperature_results.json', 'w') as temp:
                json.dump(self.temperature_data, temp)
            with open('probability_results.json', 'w') as prob:
                json.dump(self.probability_data, prob)


if __name__ == '__main__':
    rl = RobotLogger()
