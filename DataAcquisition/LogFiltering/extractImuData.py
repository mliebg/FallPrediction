# requires naobackend from HTWK Robots repo!
from naobackend.flightlogfile import FlightlogFile
from naobackend.logentryadapter import LogEntryAdapter
from naobackend.gc_logentry_parser import parse_robo_cup_game_control_data
from naobackend.proto.sensorframe_pb2 import SensorFrame as SensorFrameProto

file_to_read = "flightlog.log.z"  # replace with log file name
export_csv_name = "output.csv"  # replace with output file name

# open a flight log
print("[debug] load file")
log = FlightlogFile(file_to_read)


def filter_sd(log_entry: LogEntryAdapter):
    return log_entry.subsystem() == "SensorData" and log_entry.typeinfo() == "SensorFrame"


def log_to_sensor_frame_proto(log_entry: LogEntryAdapter) -> SensorFrameProto:
    pb = SensorFrameProto()
    pb.ParseFromString(log_entry.logEntry())
    return pb


def extract_rel_data(log_entry: LogEntryAdapter):
    return (log_entry.timestamp(),parse_robo_cup_game_control_data(log_entry.logEntry()))


print("[debug] apply filter")
sd_entries = list(map(log_to_sensor_frame_proto, log.filter(filter_sd)))
print(len(sd_entries))

print("[debug] fill csv")
f = open(export_csv_name, "w")
f.write('timestamp,gyroYaw,gyroPitch,gyroRoll,accelX,accelY,accelZ,bodyPitch,bodyRoll\n')
for e in sd_entries:
    f.write('%d,%f,%f,%f,%f,%f,%f,%f,%f\n' % (e.time, e.gyro.yaw, e.gyro.pitch, e.gyro.roll, e.accel.x, e.accel.y, e.accel.z, e.body_pitch, e.body_roll))

f.close()
