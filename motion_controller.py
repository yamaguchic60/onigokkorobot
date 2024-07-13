#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import hsrb_interface
import rospy
from sensor_msgs.msg import Image
from taggame.msg import PersonInfo  # カスタムメッセージ型をインポート

from utils.control.end_effector_wrapper import GripperWrapper
from utils.control.joint_group_wrapper import JointGroupWrapper
from utils.control.mobile_base_wrapper import MobileBaseWrapper

from utils.robot import Robot

class ExampleNode:
    def __init__(self):

        # ロボット制御部分の初期化処理
        try:
            robot = hsrb_interface.Robot()
        except Exception as e:
            rospy.logerr(f"Failed to connect to robot: {e}")
            return
        try:
            self.omni_base = MobileBaseWrapper(robot.get('omni_base'))
            rospy.loginfo("Connecting to robot 1/4")
            self.whole_body = JointGroupWrapper(robot.get('whole_body'))
            rospy.loginfo("Connecting to robot 2/4")
            gripper = GripperWrapper(robot.get('gripper'))
            rospy.loginfo("Connecting to robot 3/4")
            #self.tts = self.robot.get('default_tts') # i use robot.speak 
            rospy.loginfo("Connected!")
        except Exception as e:
            rospy.logerr(f"Failed to initialize robot components: {e}")
            return
        self.robot = Robot(robot, self.omni_base, self.whole_body, gripper)

        # PersonInfoトピックをサブスクライブ
        self.person_info_sub = rospy.Subscriber('/person_info', PersonInfo, self.person_info_callback)
        self.current_target = None
        self.last_target = None
        
        self.status = 'search'
        
        self.img_width = 640  # 画像の幅（仮定）
        self.img_height = 480  # 画像の高さ（仮定）    
            
        # モデル予測制御のパラメータ
        self.alpha = 1.0  # 距離コストの重み
        self.beta = 0.1  # 制御入力コストの重み
        self.gamma = 10.0  # 障害物回避のコストの重み
        self.prediction_horizon = 10  # 予測ホライズンの長さ
        self.control_space = np.linspace(-1.0, 1.0, 5)  # 制御入力の候補（速度の範囲）
        self.max_velocity = 0.3  # 最大速度
        self.previous_time = rospy.Time.now()
        self.obstacles = []  # 障害物のリスト douteniirerunokawakartanai
        
        
    # def image_callback(self, msg):
    #     if self.img_width is None or self.img_height is None:
    #         self.img_width = msg.width
    #         self.img_height = msg.height
    #         rospy.loginfo(f"Image size set to {self.img_width}x{self.img_height}")

        
    def person_info_callback(self, data):
        # PersonInfoメッセージを受け取ったときの処理
        self.current_target = data
        # rospy.loginfo(f"Received PersonInfo: x={data.ix}, y={data.iy}, idepth={data.idepth}, vx={data.vx}, vy={data.vy}, smile={data.smile}")
        # atodeON
    def update_status(self, current_status):
        if self.status == current_status:
            return False
        else:
            self.status = current_status
            rospy.loginfo(self.status)
            return True

    def time_diff(self):
        current_time = rospy.Time.now()
        time_diff = current_time - self.previous_time
        self.previous_time = current_time 
        
        return time_diff.to_sec()       

    def track_target_Pcontrol(self, target):

        
        if self.update_status('track'):
            self.robot.speak("I found you!",wait = True)
            

               
        # 画像の中心からの偏差を計算する
        deviation_ix =0.0 
        if target.ix and target.iy:
            deviation_ix = target.ix - self.img_width / 2
            deviation_iy = target.iy - self.img_height / 2

        # 偏差に基づいてロボットを制御する（簡単な比例制御）:P control 
        linear_speed = 0.0
        angular_speed = -0.02 * deviation_ix

        if abs(deviation_ix) < self.img_width * 0.4:
            linear_speed = target.idepth/10000
        
        time_diff = max(0.0, self.time_diff())
        
        x ,yaw = linear_speed * time_diff, angular_speed * time_diff
        rospy.loginfo(f"x = {x},yaw = {yaw},time_diff = {time_diff}")
        
        self.omni_base.go_rel(x=x, y=0.0, yaw=yaw,wait = True)
    
    
    

    def image_to_robot_coordinates(self,target):
        #ix,iy,idepth ; image's coordinates 
        #rx,ry,rz; robot's coordinate vx,vy,vz is verocity of rx,ry,rz
        rx = target.idepth * np.sin(np.arccos(target.ix/target.idepth))
        ry = - target.ix
        rz = target.iy
        return [rx,ry,rz]

    def track_target_MPCcontrol(self,target):
        if self.update_status('track'):
            self.robot.speak("I found you!", wait=True)

        [target_rx,target_ry,target_rz] = self.image_to_robot_coordinates(target)
        target_position = np.array([target_rx,target_ry])
        target_velocity = np.array([target.vx, target.vy])
        
        robot_position = np.array([0.0, 0.0])  # ロボットの現在位置（相対座標系で原点）
        best_cost = 100000000
        best_control = None

        # 予測ホライズン内での最適な制御入力を探索
        for control_x in self.control_space:
            for control_y in self.control_space:
                control = np.array([control_x, control_y])
                cost = 0.0
                robot_future_position = robot_position.copy()
                for t in range(self.prediction_horizon):
                    robot_future_position += control * t
                    target_future_position = target_position + target_velocity * t
                    cost += self.alpha * np.linalg.norm(robot_future_position - target_future_position) ** 2
                    cost += self.beta * np.linalg.norm(control) ** 2
                    for obstacle in self.obstacles:
                        cost += self.gamma / (np.linalg.norm(robot_future_position - np.array(obstacle)) + 1e-6)
                
                if cost < best_cost:
                    best_cost = cost
                    best_control = control
                    
            if best_control is not None:
                time_diff = self.time_diff()
                x = best_control[0]* time_diff
                y = best_control[1] * time_diff
                # 回転速度を計算
                angle_to_person = 0.1*np.arctan2(target_position[1], target_position[0])
                
                self.omni_base.go_rel(x,y,yaw = angle_to_person,wait = False)
        
    def search(self, last_target):
        if self.update_status('search'):
            self.robot.speak("I can't see you" , wait = True)
        angular_speed = 0.5 # 0.5 rad/s ?? adjust later
        
        if not self.last_target: last_target_direction = 1
        else:
            deviation_ix =0.0
            deviation_ix = self.last_target.ix - self.img_width /2
            if deviation_ix == 0.0:
                deviation_ix += 0.01
            last_target_direction = np.sign(deviation_ix)
            
        time_diff = self.time_diff()
        yaw = angular_speed * last_target_direction * time_diff * 0.01
        rospy.loginfo(f"Attempting to rotate: {yaw} radians")        
        
        #self.omni_base.go_abs(2.0,1.0,0.0,frame_id=world) # False -> did not move
        self.omni_base.go_rel(0.0,0.0,yaw,wait=True)#asdfasdfasdfasdfasdf
        #self.robot.speak("where are you?", wait = True)

        
    def touch(self,target):
        if not self.current_target: return False
        
        try:
            #self.whole_body.move_end_effector_pose([geometry.pose(z = 1.0),geometry.pose(z= 0.8)],ref_frame_id = 'hand_palm_link') ### NEED TO TEST!!!!!!!!!!!! z = 1.0 or 0.8 
            self.whole_body.move_to_joint_positions({'arm_lift_joint':0.3,'arm_flex_joint':-np.pi/2,'wrist_flex_joint' : 0}) # arm_lift_joint;0-0.69 wrist_flex_joint ; 70deg madesikamuri
            
            #self.whole_body.move_end_effector_by_line((0,0,1),0.3)
            
            
            # 本当にタッチしたかわわかりません。
            self.robot.speak("touch! gameover",wait = True)
            rospy.loginfo("TOUCHED GAMEOVER")
            return True # touch succeed
        
        except Exception as e:
            rospy.logerr(f"Error during touch attempt: {str(e)}")
            return False  # エラーによりタッチ失敗
        
    def main(self):
        rate = rospy.Rate(10)  # 10Hz
        TOUCH_DISTANCE = 1800 # TEST!!!!!!!!!!!!!!!!!!!!!! #############
        
        self.robot.speak("Let's play tag game together!")
        rospy.loginfo(f"Subscribed to topic: {self.person_info_sub.name}")

        self.whole_body.move_to_go()
        self.omni_base.go_abs(2.0,1.0,0.0,frame_id='world') # start position
        self.whole_body.move_to_joint_positions({'head_tilt_joint': 0.0}) 
        self.robot.speak('start',wait=True)
             
        while not rospy.is_shutdown():
            self.whole_body.move_to_joint_positions({'head_tilt_joint': 0.0}) 
            
            rospy.loginfo(f"Current status: {self.status}")
            #rospy.loginfo(self.current_target) #ok
            try:
                if not self.current_target:
                    #rospy.loginfo("searching")
                    self.search(self.last_target)
                    print("search")
                    
                else:
                    #rospy.loginfo(f"Target detected at depth: {self.current_target.idepth}")
                    # deviation_ix = 0.0
                    # deviation_ix = self.current_target.ix - self.img_width /2
                    if self.current_target.idepth < TOUCH_DISTANCE :#and abs(deviation_ix) < self.img_width * 0.3: 
                        if self.touch(self.current_target):
                            break
                    else:
                        print("trach target control")
                        self.track_target_Pcontrol(self.current_target)
                        self.last_target = self.current_target
                        
            except Exception as e:
                self.robot.speak(f"error : {e}. STOP")
                rospy.logerr(f"error : {e}")
                break
                
            rate.sleep()

if __name__ == '__main__':
    example_node = ExampleNode()
    if hasattr(example_node, 'robot'):
        example_node.main()
    rospy.spin()


            
    
    
    
        

