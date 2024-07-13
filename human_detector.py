#!/usr/bin/env python3

# pip install ultralytics

# rosrun robocup_utils human_detector.py
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
from collections import defaultdict
from scipy.spatial import distance
from taggame.msg import PersonInfo  # 新しいメッセージ型をインポート

from ultralytics import YOLO
import base64
import requests

class HumanDetector:
    def __init__(self):
        rospy.init_node('human_detector', anonymous=True)

        # Publisher
        self.person_info_pub = rospy.Publisher('/person_info', PersonInfo, queue_size=10)

        # Subscribers
        self.image_sub = Subscriber('/relay/head/rgb/aligned', Image)
        self.depth_sub = Subscriber('/relay/head/depth/aligned', Image)
        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_depth_callback)

        self.bridge = CvBridge()

        # YOLOv5モデルの読み込み
        self.model = None
        try:
            self.model = YOLO('yolov8n.pt')
            rospy.loginfo("YOLOv8 model loaded successfully")
        except Exception as e:
            rospy.logerr("Failed to load YOLOv8 model: %s", str(e))
            

        self.previous_positions = {}
        self.previous_features = {}
        self.previous_time = rospy.Time.now()
        self.detected_persons = defaultdict(int)
        self.person_indices = {}
        self.last_detection_time = rospy.Time.now()
        self.detection_threshold = rospy.get_param('~detection_threshold', 3)
        self.timeout_duration = rospy.get_param('~timeout_duration', 5)  # seconds


        self.api_key = "your api key"

    def coordinate_tf(self,ix,iy,idepth):
        #ix,iy,idepth ; image's coordinates 
        #rx,ry,rz; robot's coordinate vx,vy,vz is verocity of rx,ry,rz
        rx = idepth * np.sin(np.arccos(ix/idepth))
        ry = - ix
        rz = iy
        return rx,ry,rz
                

    def image_depth_callback(self, image_data, depth_data):

        if self.model is None:
            rospy.logerr("YOLOv8 model not loaded. Skipping detection.")
            return
        frame = self.bridge.imgmsg_to_cv2(image_data, 'bgr8')
        depth_frame = self.bridge.imgmsg_to_cv2(depth_data, '32FC1')

        results = self.model(frame, verbose=False)
        persons = results[0].boxes[results[0].boxes.cls == 0] # Class 0 corresponds to 'person'
        current_time = rospy.Time.now()

        person_info_list = []
        current_positions = []

        for person in persons:
            ix1, iy1, ix2, iy2, conf, cls = person.data[0].tolist()
            ix1, iy1, ix2, iy2 = int(ix1), int(iy1), int(ix2), int(iy2)
            person_center_ix = (ix1 + ix2) // 2
            person_center_iy = (iy1 + iy2) // 2

            # 距離の取得
            person_depth = depth_frame[person_center_iy, person_center_ix]

            # 現在の検出位置を保存
            current_positions.append((person_center_ix, person_center_iy))

            # 検出した人物の領域
            person_roi = frame[iy1:iy2, ix1:ix2]
            cv2.imwrite(f"detected_img.jpg", person_roi)

            # calcurate the smile rate
            smile_rate = self.smile_counter(person_roi)
            
            # 服の色特徴の取得
            avg_color = np.mean(person_roi, axis=(0, 1))

            # 以前の検出位置と比較して同一人物かを判定
            matched_index = self.match_person(person_center_ix, person_center_iy, avg_color)

            #  速度の計算
            if matched_index in self.previous_positions:
                previous_position = self.previous_positions[matched_index]
                time_diff = (current_time - self.previous_time).to_sec()
                #ix,iy,idepth ; image's coordinates 
                # rx,ry,rz; robot's coordinate vx,vy,vz is verocity of rx,ry,rz
                rx,ry,rz =self.coordinate_tf(person_center_ix,person_center_iy,person_depth)
                prev_rx,prev_ry,prev_rz = self.coordinate_tf(previous_position[0],previous_position[1],previous_position[2])
                velocity_rx = (rx - prev_rx) / time_diff
                velocity_ry = (ry - prev_ry) / time_diff
            else:
                velocity_rx, velocity_ry = 0, 0

            # 画像に表示
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)
            cv2.circle(frame, (person_center_ix, person_center_iy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f'ID: {matched_index}, Depth: {person_depth:.2f}m', (ix1, iy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Vel: ({velocity_rx:.2f}, {velocity_ry:.2f})', (ix1, iy1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # データの保存
            self.previous_positions[matched_index] = (person_center_ix, person_center_iy, person_depth)
            self.previous_features[matched_index] = avg_color
            person_info = PersonInfo()
            person_info.ix = person_center_ix
            person_info.iy = person_center_iy
            person_info.idepth = float(person_depth)
            person_info.vx = velocity_rx
            person_info.vy = velocity_ry
            person_info.smile = smile_rate
            person_info_list.append(person_info)

        # 検出時間の更新
        if persons:
            self.last_detection_time = current_time

        # 画像の表示
        #cv2.imshow('Person Detection', frame)
        #cv2.waitKey(1)

        # データのパブリッシュ
        smile_counter = 0
        smilest_person_list = None
        if person_info_list:
            for info in person_info_list:
                if int(info.smile) > int(smile_counter):
                    smilest_person_list = info
                    smile_counter = info.smile
            if smilest_person_list:
                self.person_info_pub.publish(smilest_person_list)
                rospy.loginfo(f"Published PersonInfo: ix={smilest_person_list.ix}, iy={smilest_person_list.iy}, idepth={smilest_person_list.idepth}, vx={smilest_person_list.vx}, vy={smilest_person_list.vy}, smile={smilest_person_list.smile}")

        self.previous_time = current_time

        # タイムアウトによるリセット
        if (current_time - self.last_detection_time).to_sec() > self.timeout_duration:
            self.previous_positions.clear()
            self.previous_features.clear()
            self.detected_persons.clear()
            self.person_indices.clear()

    def match_person(self, x, y, avg_color):
        min_distance = float('inf')
        matched_index = None
        for idx, (prev_x, prev_y, prev_depth) in self.previous_positions.items():
            pos_distance = np.linalg.norm([x - prev_x, y - prev_y])
            color_distance = distance.euclidean(avg_color, self.previous_features[idx])
            total_distance = pos_distance + color_distance
            if total_distance < min_distance and total_distance < 50:  # Threshold for matching
                min_distance = total_distance
                matched_index = idx

        if matched_index is not None:
            self.detected_persons[matched_index] += 1
            if self.detected_persons[matched_index] >= self.detection_threshold:
                return matched_index

        new_index = len(self.person_indices) + 1
        self.person_indices[new_index] = (x, y)
        self.previous_features[new_index] = avg_color
        self.detected_persons[new_index] = 1
        return new_index

    def encode_image_np(self, image_np):
        _, buffer = cv2.imencode('.jpg', image_np)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64

    def smile_counter(self, person_roi):
        encoded_img = self.encode_image_np(person_roi)

        # Prepare payload for OpenAI API
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Rate the degree of fun of the person you observed on a scale of 0 to 10. The standard value is 5. Please judge the rating based on the following factors: First, observe the person's smile. A big smile, wrinkles at the corners of the eyes, and a smile that shows teeth indicates a high level of fun. Conversely, a blank expression or a slight smile indicates a low level of fun. Next, observe the person's movements. If the person is actively moving around, jumping and waving energetically, the level of fun is high. If there is little movement and the movements are sluggish, the level of fun is low. Taking these factors into consideration, rate how fun the person you are observing looks on a scale of 0 to 10. Return only the value absolutely."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_img}"
                        }
                    }
                ],
            }
        ]

        payload = {
            "model": "gpt-4o",
            "messages": messages,
            "max_tokens": 1
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        response_json = response.json()
        message_content = response_json['choices'][0]['message']['content']

        try:
            smile_rate = float(message_content)
        except ValueError:
            smile_rate = 5.0

        return smile_rate


    def process(self):
        rospy.spin()

if __name__ == '__main__':
    human_detector = HumanDetector()
    try:
        human_detector.process()
    except rospy.ROSInterruptException:
        pass

