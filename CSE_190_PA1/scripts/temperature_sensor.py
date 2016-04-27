#!/usr/bin/env python

import rospy
import random
import math as m
import numpy as np
from copy import deepcopy
from cse_190_assi_1.srv import requestMapData
from cse_190_assi_1.msg import temperatureMessage
from std_msgs.msg import Bool
from read_config import read_config


class TempSensor():
    def __init__(self):
        """Read config file and setup ROS things"""
        self.config = read_config()
        rospy.init_node("temperature_sensor")
        self.temperature_requester = rospy.ServiceProxy(
                "requestMapData",
                requestMapData
        )
        self.activation_service = rospy.Subscriber(
                "/temp_sensor/activation",
                Bool,
                self.handle_activation_message
        )
        self.temperature_publisher = rospy.Publisher(
                "/temp_sensor/data",
                temperatureMessage,
                queue_size = 10
        )
        self.temp_dict = {
                'H': 40.0,
                'C': 20.0,
                '-': 25.0
        }
        self.temp_message = temperatureMessage()
        self.sensor_on = False
        random.seed(self.config['seed'])
        self.sensor_loop()

    def handle_activation_message(self, message):
        """Callback function for the activation subscriber.
        
        If the incoming message has a value of True, the node begins
        publishing temperature data at a rate of 1Hz.
        """
        #print "incoming message, ", message
        switch_value = message.data
        self.sensor_on = switch_value
        if switch_value:
            self.rate = rospy.Rate(1)

    def sensor_loop(self):
        """Publish a noisy temperature measurement every second."""
        while not rospy.is_shutdown():
            if self.sensor_on:
                measurement = self.take_measurement()
                noisy_measurement = self.add_noise(measurement)
                self.temp_message.temperature = noisy_measurement
                self.temperature_publisher.publish(self.temp_message)
                self.rate.sleep()

    def take_measurement(self):
        """Get the temperature of the current square."""
        temp_response = self.temperature_requester('temp')
        temperature = temp_response.data
        temp = self.temp_dict[temperature]
        return temp

    def add_noise(self, true_val):
        """Returns measurement after adding Gaussian noise."""
        noise = m.ceil(random.gauss(0, self.config['temp_noise_std_dev'])*100.)/100.
        noisy_measurement = true_val + noise
        return noisy_measurement


if __name__ == '__main__':
    ts = TempSensor()
