#Roboter gilt als isFallen ab 25° 
# is_fallen = abs(angles.pitch) > fall_threshold || abs(angles.roll) > fall_threshold;
# => https://naoserv.imn.htwk-leipzig.de/redmine/projects/nao/repository/2/entry/firmware_5.0/motion/getup_controller.cpp
# static constexpr float fall_threshold = 25_deg; 
# => https://naoserv.imn.htwk-leipzig.de/redmine/projects/nao/repository/2/entry/firmware_5.0/motion/getup_controller.h
#Body Angles werden aus IMU Daten berechnet
# => https://naoserv.imn.htwk-leipzig.de/redmine/projects/nao/repository/2/entry/firmware_5.0/imu/filter.cpp
import math
#import matplotlib.pyplot as plt

from naobackend import lola_status
from naobackend.flightlogfile import FlightlogFile
from naobackend.logentryadapter import LogEntryAdapter
from naobackend.lola_status import LolaStatus
from naobackend.lola_status import YPR
from naobackend.lola_status import Point3d
from  naobackend.proto.lola_lowlevel_pb2 import LolaStatus as LolaStatusProto

file_to_read = "D:/Data/Finals-B-Human-2023-07-09-14-32-35/Pathfinder_5/flightlog-2023-07-09-11-25-34.log.z"
export_csv_name = "imu_logdata_5_1.csv"

# open a flight log
print("[debug] load file")
log = FlightlogFile(file_to_read)

def filter_cb(log_entry: LogEntryAdapter):
    return log_entry.subsystem() == "LolaConnector" and log_entry.typeinfo() == "LolaDebugFrame"


def log_to_lola_status_proto(log_entry: LogEntryAdapter) -> LolaStatusProto:
    pb = LolaStatusProto()
    pb.ParseFromString(log_entry.logEntry())
    return pb

def norm(p: Point3d):
    return math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z)

def norm_sqr(p: Point3d):
    return p.x * p.x + p.y * p.y + p.z * p.z

def normalize(p: Point3d):
    ing = 1. / norm(p)
    p.x *= ing
    p.y *= ing
    p.z *= ing
    return p

def calculate_body_angles(gyro: YPR, accel: Point3d) -> lola_status.YPR:
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #Es folgt Body.Angles Berechnung nach
    # https://naoserv.imn.htwk-leipzig.de/redmine/projects/nao/repository/2/entry/firmware_5.0/imu/filter.cpp
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    invSampleFreq = 0.012   # seconds
    beta = 0.1              # 2 * proportional gain (Kp)

    #quaternion of sensor frame relative to auxiliary frame
    q0 = 1.
    q1 = 0.
    q2 = 0.
    q3 = 0.

    #Rate of change of quaternion from gyroscope
    qDot1 = 0.5 * (-q1 * gyro.roll - q2 * gyro.pitch - q3 * gyro.yaw)
    qDot2 = 0.5 * (q0 * gyro.roll + q2 * gyro.yaw - q3 * gyro.pitch)
    qDot3 = 0.5 * (q0 * gyro.pitch - q1 * gyro.yaw + q3 * gyro.roll)
    qDot4 = 0.5 * (q0 * gyro.yaw + q1 * gyro.pitch - q2 * gyro.roll)

    #Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)
    if (norm_sqr(accel) > 0.):
        #Auxiliary variables to avoid repeated arithmetic
        _2q0 = 2.0 * q0
        _2q1 = 2.0 * q1
        _2q2 = 2.0 * q2
        _2q3 = 2.0 * q3
        _4q0 = 4.0 * q0
        _4q1 = 4.0 * q1
        _4q2 = 4.0 * q2
        _8q1 = 8.0 * q1
        _8q2 = 8.0 * q2
        q0q0 = q0 * q0
        q1q1 = q1 * q1
        q2q2 = q2 * q2
        q3q3 = q3 * q3
        # Gradient decent algorithm corrective step
        # Madgwick's original implementation assumes the z axis goes towards the earth. Our coordinate system defines z
        # in the exact opposite direction. Rotating the reference vector (formula 23 in the original paper) by 180
        # degrees inverts all terms containing accel values below.
        s0 = _4q0 * q2q2 - _2q2 * accel.x + _4q0 * q1q1 + _2q1 * accel.y
        s1 = _4q1 * q3q3 + _2q3 * accel.x + 4.0 * q0q0 * q1 + _2q0 * accel.y - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 - _4q1 * accel.z
        s2 = 4.0 * q0q0 * q2 - _2q0 * accel.x + _4q2 * q3q3 + _2q3 * accel.y - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 - _4q2 * accel.z
        s3 = 4.0 * q1q1 * q3 + _2q1 * accel.x + 4.0 * q2q2 * q3 + _2q2 * accel.y
        sq = s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3
        
        if(sq <= 0):
            return YPR(8484.8484,8484.8484,sq)
        recipNorm = 1. / math.sqrt(sq);  # normalize step magnitude
        s0 *= recipNorm
        s1 *= recipNorm
        s2 *= recipNorm
        s3 *= recipNorm
        # Apply feedback step
        qDot1 -= beta * s0
        qDot2 -= beta * s1
        qDot3 -= beta * s2
        qDot4 -= beta * s3
    
    #Integrate rate of change of quaternion to yield quaternion
    q0 += qDot1 * invSampleFreq
    q1 += qDot2 * invSampleFreq
    q2 += qDot3 * invSampleFreq
    q3 += qDot4 * invSampleFreq

    #Normalize quaternion
    recipNorm = 1. / math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
    q0 *= recipNorm
    q1 *= recipNorm
    q2 *= recipNorm
    q3 *= recipNorm

    angles_yaw = math.atan2(q1 * q2 + q0 * q3, 0.5 - q2 * q2 - q3 * q3);                               
    angles_pitch = math.asin(-2.0 * (q1 * q3 - q0 * q2))
    angles_roll = math.atan2(q0 * q1 + q2 * q3, 0.5 - q1 * q1 - q2 * q2)
    #forward = {2 * (q1 * q3 + q0 * q2), 2 * (q2 * q3 - q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2)};     wird nicht benötigt
    #up = {2 * (q1 * q2 - q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 - q0 * q1)};          wird nicht benötigt
    return YPR(angles_yaw,angles_pitch,angles_roll)

print("[debug] apply filter")
lola_status_entries: list[LolaStatus]
lola_status_entries = list(map(lola_status.convert_from_proto, map(log_to_lola_status_proto, log.filter(filter_cb))))

print("LolaStatusEntries Count: ")
print(len(lola_status_entries))
#print("LolaStatusEntries TS last: ")
#print(lola_status_entries[len(lola_status_entries) - 1].timestamp)

f = open(export_csv_name,"w")
f.write("timestamp,gyroYaw,gyroPitch,gyroRoll,accelX,accelY,accelZ,bodyPitch,bodyRoll,isFallen,NextTsIn0.5,d0.5,NextTsIn1.0,d1.0,NextTsIn2.0,d2.0\n")
print("[debug] filling csv")

for i in range(len(lola_status_entries) - 1):
    bodyAngles = calculate_body_angles(lola_status_entries[i].imu.gyro,lola_status_entries[i].imu.accel)

    #look forward to see which timestamp is +0.1s, +1.0s and #2.0s
    plus_dot5 = lola_status_entries[i].timestamp # + 500_000
    plus_1 = lola_status_entries[i].timestamp    # + 1_000_000
    plus_2 = lola_status_entries[i].timestamp    # + 2_000_000

    found_dot5 = False
    found_1 = False
    counter = i
    while (plus_2 < lola_status_entries[i].timestamp + 2_000_000):
        counter = counter + 1

        #es wird nicht über logfiles hinausgeschaut -> negativer ts als null ersatz
        if(counter > len(lola_status_entries) - 1):
            if(not found_dot5):
                plus_dot5 = -9999
            if(not found_1):
                plus_1 = -9999
            plus_2 = -9999
            break

        if(not found_dot5):
            if(lola_status_entries[counter].timestamp >= lola_status_entries[i].timestamp + 500_000):
                plus_dot5 = lola_status_entries[counter].timestamp
                found_dot5 = True
        if(not found_1):
            if(lola_status_entries[counter].timestamp >= lola_status_entries[i].timestamp + 1_000_000):
                plus_1 = lola_status_entries[counter].timestamp
                found_1 = True
        if(lola_status_entries[counter].timestamp >= lola_status_entries[i].timestamp + 2_000_000):
                plus_2 = lola_status_entries[counter].timestamp
                

    is_fallen = (bodyAngles.pitch > 25*math.pi/180 or bodyAngles.pitch > 25*math.pi/180)


    f.write("%d,%f,%f,%f,%f,%f,%f,%f,%f,%s,%d,%d,%d,%d,%d,%d\n"% (lola_status_entries[i].timestamp,
                                                                    lola_status_entries[i].imu.gyro.yaw,
                                                                    lola_status_entries[i].imu.gyro.pitch,
                                                                    lola_status_entries[i].imu.gyro.roll,
                                                                    lola_status_entries[i].imu.accel.x,
                                                                    lola_status_entries[i].imu.accel.y,
                                                                    lola_status_entries[i].imu.accel.z,
                                                                    bodyAngles.pitch,bodyAngles.roll,
                                                                    is_fallen,
                                                                    plus_dot5,plus_dot5-lola_status_entries[i].timestamp-500_000,
                                                                    plus_1,plus_1-lola_status_entries[i].timestamp-1_000_000,
                                                                    plus_2,plus_2-lola_status_entries[i].timestamp-2_000_000))
                                                                    

f.close()

#folge RBQL Query: 
#select a2, a3, a4, a5, a6, a7, b.isFallen LEFT JOIN ./imu_logdata_5_1.csv ON a15 == b1 order by a1 asc
#Bereinigung von nulls:
#select * where a7 != ''