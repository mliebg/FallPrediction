import random
import matplotlib.pyplot as plt

from naobackend import lola_status
from naobackend.flightlogfile import FlightlogFile
from naobackend.logentryadapter import LogEntryAdapter
from naobackend.lola_status import LolaStatus
from  naobackend.proto.lola_lowlevel_pb2 import LolaStatus as LolaStatusProto

file_to_read = "D:/Data/Finals-B-Human-2023-07-09-14-32-35/CometInterceptor_4/flightlog-2023-07-09-11-44-24.log.z"

# open a flight log
print("[debug] load file")
log = FlightlogFile(file_to_read)

def filter_cb(log_entry: LogEntryAdapter):
    return log_entry.subsystem() == "LolaConnector" and log_entry.typeinfo() == "LolaDebugFrame"


def log_to_lola_status_proto(log_entry: LogEntryAdapter) -> LolaStatusProto:
    pb = LolaStatusProto()
    pb.ParseFromString(log_entry.logEntry())
    return pb

print("[debug] apply filter")
lola_status_entries: list[LolaStatus]
lola_status_entries = list(map(lola_status.convert_from_proto, map(log_to_lola_status_proto, log.filter(filter_cb))))

print("[debug] fill lists")
timestamps = list(map(lambda x: x.timestamp, lola_status_entries)) 
gyroYaws = list(map(lambda x: x.imu.gyro.yaw, lola_status_entries))
gyroPitchs = list(map(lambda x: x.imu.gyro.pitch, lola_status_entries))
gyroRolls = list(map(lambda x: x.imu.gyro.roll, lola_status_entries))
accelXs = list(map(lambda x: x.imu.accel.x, lola_status_entries))
accelYs = list(map(lambda x: x.imu.accel.y, lola_status_entries))
accelZs = list(map(lambda x: x.imu.accel.z, lola_status_entries))
'''
print("yaws: ")
print(len(gyroYaws) )
print("pitchs: ")
print(len(gyroPitchs))
print("rolls: ")
print(len(gyroRolls))
print("accelXs: ")
print(len(accelXs))
print("accelYs: ")
print(len(accelYs))
print("accelZs: ")
print(len(accelZs))
'''

f = open("imu_logdata_4_random.csv","w")
f.write("timestamp,gyroYaw,gyroPitch,gyroRoll,accelX,accelY,accelZ,isFallen\n")
print("[debug] filling csv")
for i in range(len(gyroYaws)):
    if random.random() <= .01 :
        f.write("%d,%f,%f,%f,%f,%f,%f,true\n"% (timestamps[i],gyroYaws[i],gyroPitchs[i],gyroRolls[i],accelXs[i],accelYs[i],accelZs[i]))
    else :
        f.write("%d,%f,%f,%f,%f,%f,%f,false\n"% (timestamps[i],gyroYaws[i],gyroPitchs[i],gyroRolls[i],accelXs[i],accelYs[i],accelZs[i]))
#print(battery)
#plt.plot(battery)
#plt.show()

