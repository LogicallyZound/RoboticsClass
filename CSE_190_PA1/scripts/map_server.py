#!/usr/bin/env python

import rospy
import random as r
import math as m
import numpy as np
from copy import deepcopy
from cse_190_assi_1.srv import requestMapData, moveService
from read_config import read_config


class MapServer():
    def __init__(self):
        """Read config file and setup ROS things"""
        self.config = read_config()
        self.config["prob_move_correct"] = .75
        rospy.init_node("map_server")
        self.map_data_service = rospy.Service(
                "requestMapData",
                requestMapData,
                self.handle_data_request
        )
        self.move_service = rospy.Service(
                "moveService",
                moveService,
                self.handle_move_request
        )
        self.pos = self.initialize_position()
        r.seed(self.config['seed'])
        print "starting pos: ", self.pos
        rospy.spin()

    def initialize_position(self):
        """Set starting position."""
        pos = self.config["starting_pos"]
        return pos

    def handle_data_request(self, request):
        """Service to provide ground truth data to sensors"""
        if request.data_type == "temp":
            temp = self.config['pipe_map'][self.pos[0]][self.pos[1]]
            return temp
        if request.data_type == "tex":
            tex = self.config['texture_map'][self.pos[0]][self.pos[1]]
            return tex

    def handle_move_request(self, request):
        """Service that moves the robot according to a move request.

        self.config['uncertain_motion'] determines the motion model
        used: either certain motion or motion with a set probability
        (randomotherwise).
        """
        move = list(request.move)
        if self.config['uncertain_motion']:
            roll = r.uniform(0,1)
            if roll < self.config["prob_move_correct"]:
                self.make_move(move)
            else:
                possible_moves = deepcopy(self.config['possible_moves'])
                possible_moves.remove(move)
                random_move = r.choice(possible_moves)
                self.make_move(random_move)
        elif not self.config['uncertain_motion']:
            self.make_move(move)
        return []

    def make_move(self, move):
        """Changes the robot's position"""
        num_rows = len(self.config['pipe_map'])
        num_cols = len(self.config['pipe_map'][0])
        self.pos[0] = (self.pos[0] + move[0]) % num_rows
        self.pos[1] = (self.pos[1] + move[1]) % num_cols
        #print self.pos


if __name__ == '__main__':
    ms = MapServer()
