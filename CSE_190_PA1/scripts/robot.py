#!/usr/bin/env python
import rospy
import math
import itertools
import copy
from std_msgs.msg import Bool, String, Float32
from cse_190_assi_1.msg import temperatureMessage, RobotProbabilities
from cse_190_assi_1.srv import requestTexture, moveService
from read_config import read_config

class robot():
   def __init__(self):

      rospy.init_node('robot_node', anonymous = True)

      self.temp_dict = { 'C' : 20,
                         '-' : 25,
                         'H' : 40 }

      self.move_dict = { tuple([0,0]) : "hold",
                         tuple([1,0]) : "down",
                         tuple([-1,0]): "up",
                         tuple([0, 1]): "right",
                         tuple([0,-1]): "left" }

      self.config    = read_config()
      self.sig       = self.config["temp_noise_std_dev"]
      self.p_tex     = self.config["prob_tex_correct"]
      self.p_mov     = self.config["prob_move_correct"]
      self.move_list = self.config["move_list"]
      self.temp_map  = self.config["pipe_map"]
      self.tex_map   = self.config["texture_map"]
      self.height    = len(self.config["pipe_map"])
      self.width     = len(self.config["pipe_map"][0])
      self.fail_mov  = (1-self.p_mov)/4
        
      #The current and maximum number of moves we have
      self.move_count = 0 
      self.move_size  = len(self.move_list)

      #create initial belief grid
      initial      = 1./(self.height*self.width)
      self.beliefs = []
      for i in range(0, self.height):
         self.beliefs.append( [initial]*self.width )

      #services
      self.texProxy = rospy.ServiceProxy('requestTexture', requestTexture)
      self.movProxy = rospy.ServiceProxy('moveService', moveService)
      self.temp_sub = rospy.Subscriber("/temp_sensor/data", 
                                 temperatureMessage, self.handle_incoming_temp)
      publisher     = rospy.Publisher("/temp_sensor/activation",   Bool,               queue_size = 10)
      self.closepub = rospy.Publisher("/map_node/sim_complete",    Bool,               queue_size = 10)
      self.prob_pub = rospy.Publisher("/results/probabilities",    RobotProbabilities, queue_size=10)
      self.temp_pub = rospy.Publisher("/results/temperature_data", Float32,            queue_size=10)
      self.tex_pub  = rospy.Publisher("/results/texture_data",     String,             queue_size = 10)

      #Activate the temp sensor and keep robot node up
      rospy.sleep(3)
      publisher.publish(True)
      rospy.spin()

   #actual is T(xi)
   def guassian_eq_temp(self, sigma, real, actual):
      return (1 / (sigma*math.sqrt(2*math.pi))) *math.exp( (-(real - actual )**2)/(2*sigma**2)  )

   def normalize_grid(self, normalizer):
      for row in range(len(self.beliefs)):
         for col in range(len(self.beliefs[row])):
            self.beliefs[row][col] =  self.beliefs[row][col] / normalizer

   def update_temperature(self, temp):
      normal = 0

      for row in range(len(self.beliefs)):
         for col in range(len(self.beliefs[row])):
            #temperature bayes rule here
            actual_temp     = self.temp_dict[self.temp_map[row][col]]
            p_temp_given_Xi = self.guassian_eq_temp(self.sig, temp, actual_temp) 
            prior_Xi        = self.beliefs[row][col]
            numerator       = p_temp_given_Xi * prior_Xi

            #sum the normalization constant for division at the end
            normal                += numerator
            self.beliefs[row][col] = numerator

      #normalize at the end when we have our normalization factor
      self.normalize_grid(normal)

   def get_texture_and_update(self):
      texReading = self.texProxy()
      tex = texReading.data 

      normal = 0
      for row in range(len(self.beliefs)):
         for col in range(len(self.beliefs[row])):
            p_tex_given_Xi = self.p_tex if tex == self.tex_map[row][col] else 1-self.p_tex
            prior_Xi = self.beliefs[row][col] 
            numerator = p_tex_given_Xi * prior_Xi
            #sum the normalization constant for division at the end
            normal += numerator
            self.beliefs[row][col] = numerator
      #normalize at the end
      self.normalize_grid(normal)
      return tex

      #print "got texture: ", texData

   def move_and_update(self):
      #move as list
      move = self.move_list[self.move_count]
      #move as string
      move_str = self.move_dict[tuple(move)]
      self.movProxy(move)

      old_beliefs = copy.deepcopy(self.beliefs) 
      for row in range(len(self.beliefs)):
         for col in range(len(self.beliefs[row])):
            if move_str == "hold" :
               self.beliefs[row][col] = (self.p_mov*old_beliefs[row][col] + 
                                         self.fail_mov*old_beliefs[(row+self.height-1)%self.height][col]+
                                         self.fail_mov*old_beliefs[(row+self.height+1)%self.height][col]+
                                         self.fail_mov*old_beliefs[row][(col+self.width+1)%self.width] + 
                                         self.fail_mov*old_beliefs[row][(col+self.width-1)%self.width] ) 
            elif move_str == "down" :
               self.beliefs[row][col] = (self.fail_mov*old_beliefs[row][col] + 
                                         self.p_mov*old_beliefs[(row+self.height-1)%self.height][col]+
                                         self.fail_mov*old_beliefs[(row+self.height+1)%self.height][col]+
                                         self.fail_mov*old_beliefs[row][(col+self.width+1)%self.width] + 
                                         self.fail_mov*old_beliefs[row][(col+self.width-1)%self.width] ) 
            elif move_str == "right" :
               self.beliefs[row][col] = (self.fail_mov*old_beliefs[row][col] + 
                                         self.fail_mov*old_beliefs[(row+self.height-1)%self.height][col]+
                                         self.fail_mov*old_beliefs[(row+self.height+1)%self.height][col]+
                                         self.fail_mov*old_beliefs[row][(col+self.width+1)%self.width] + 
                                         self.p_mov*old_beliefs[row][(col+self.width-1)%self.width] ) 
            elif move_str == "left" :
               self.beliefs[row][col] = (self.fail_mov*old_beliefs[row][col] + 
                                         self.fail_mov*old_beliefs[(row+self.height-1)%self.height][col]+
                                         self.fail_mov*old_beliefs[(row+self.height+1)%self.height][col]+
                                         self.p_mov*old_beliefs[row][(col+self.width+1)%self.width] + 
                                         self.fail_mov*old_beliefs[row][(col+self.width-1)%self.width] ) 
            elif move_str == "up" :
               self.beliefs[row][col] = (self.fail_mov*old_beliefs[row][col] + 
                                         self.fail_mov*old_beliefs[(row+self.height-1)%self.height][col]+
                                         self.p_mov*old_beliefs[(row+self.height+1)%self.height][col]+
                                         self.fail_mov*old_beliefs[row][(col+self.width+1)%self.width] + 
                                         self.fail_mov*old_beliefs[row][(col+self.width-1)%self.width] ) 
      self.move_count  += 1

   def publish_beliefs(self):
      flatten_beliefs = itertools.chain(*self.beliefs)
      self.prob_pub.publish(list(flatten_beliefs))

   def publish_temperature(self, temp):
      self.temp_pub.publish(temp)
      
   def publish_texture(self, tex):
      self.tex_pub.publish(tex)

   def handle_shutdown(self):
      self.closepub.publish(True)
      rospy.sleep(3)
      rospy.signal_shutdown("No more moves")

   def can_move(self):
      return self.move_count < self.move_size

   def handle_incoming_temp(self, message):
      self.update_temperature(message.temperature)
      texture = self.get_texture_and_update()
      
      self.publish_temperature(message.temperature)
      self.publish_texture(texture)

      if self.can_move():
         self.move_and_update()
         self.publish_beliefs()

      else:
         self.publish_beliefs()
         self.handle_shutdown()


if __name__ == "__main__":
   ln = robot() 
