#!/usr/bin/env python

import rospy
import random as r
import math as m
import numpy as np
from copy import deepcopy
from cse_190_assi_1.srv import requestMapData, requestTexture 
from read_config import read_config


class TexSensor():
    def __init__(self):
        """Read config file and setup ROS things"""
        self.config = read_config()
        rospy.init_node("texture_sensor")
        self.texture_requester = rospy.ServiceProxy(
                "requestMapData",
                requestMapData
        )
        self.texture_service = rospy.Service(
                "requestTexture",
                requestTexture,
                self.handle_texture_request
        )
        r.seed(self.config['seed'])
        rospy.sleep(1)
        rospy.spin()

    def handle_texture_request(self, request):
        """Callback function for the texture service."""
        texture = self.take_measurement()
        noisy_texture = self.add_noise(texture)
        return noisy_texture

    def take_measurement(self):
        """Get the texture of the current square."""
        tex_response = self.texture_requester('tex')
        tex = tex_response.data
        return tex

    def add_noise(self, measurement):
        """Flip measurement with some probability."""
        roll = r.uniform(0,1)
        if roll < self.config['prob_tex_correct']:
            return measurement
        else:
            if measurement == 'R':
                return 'S'
            elif measurement == 'S':
                return 'R'


if __name__ == '__main__':
    ts = TexSensor()
