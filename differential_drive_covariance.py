import numpy as np
import math

#Reference
# https://github.com/Sollimann/CleanIt/blob/main/autonomy/src/slam/README.md 

class errorModel():
    
    def __init__(self,x=0,y=0,yaw=0,s1=0,s2=0):
        
        self.wheel_dist = 0.67
        self.x = x         # x coordinate of the robot
        self.y = y         # y coordinate 
        self.yaw = yaw     # yaw of the robot
        self.s1 = s1       # the distance travelled by wheel 1
        self.s2 = s2       # the distance travelled by wheel 2
        self.covariance = np.full(shape = (3,3),fill_value = 0.00001) # initial covariance
    
    def covariance_calc(self,s1,s2):  #calculates the covariance 
        self.update(s1,s2)  
        vel_cov = self.velocity_jacobian.dot(self.motion_cov).dot(np.transpose(self.velocity_jacobian))
        final_cov = self.pose_jacobian.dot(self.covariance).dot(np.transpose(self.pose_jacobian))
        final_cov += vel_cov
        self.covariance = final_cov
        
    
    def update(self,s1,s2): # updates x,y,yaw and the jacobians using s1 and s2 
        self.s1 = s1
        self.s2 = s2
        
        #Differential drive equations 
        ds = (self.s1+self.s2)/2
        dyaw = (self.s1 - self.s2)/(2*self.wheel_dist)
        cos_yaw = math.cos(self.yaw + (dyaw/2))
        sin_yaw = math.sin(self.yaw + (dyaw/2))
        dx = ds*cos_yaw
        dy = ds*sin_yaw
        
        #updating the coordinates
        self.yaw += dyaw
        self.x += dx
        self.y += dy
        k1 = 0.1 # experimental constant for error in s1
        k2 = 0.1 # experimental constant for error in s2
        
        #updating the values of the matrices
        self.motion_cov = np.array([
                        [k1*abs(s1),0],
                        [0,k2*abs(s2)]
                    ])
                        
        self.velocity_jacobian = np.array([
                        [(cos_yaw/2) - ((ds/(2*self.wheel_dist))*sin_yaw),(cos_yaw/2) + ((ds/(2*self.wheel_dist))*sin_yaw)],
                        [(sin_yaw/2) + ((ds/(2*self.wheel_dist))*cos_yaw),(sin_yaw/2) - ((ds/(2*self.wheel_dist))*cos_yaw)],
                        [1/self.wheel_dist,-1/self.wheel_dist],
                    ])
                        
        self.pose_jacobian = np.array([
                        [1,0,-dy],
                        [0,1, dx],
                        [0,0,  1]
                    ])
        
#example driver code       
er = errorModel()
s1 = [0.1,0.3,0.2,0.1]
s2 = [0.1,0.2,0.1,0.3]
for i in range(len(s1)):
    er.covariance_calc(s1[i],s2[i])

print(er.covariance)     
