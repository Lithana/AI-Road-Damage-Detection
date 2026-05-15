import serial
import time

def get_hardware_gps():
    """
    Example function to read from a Neo-6M GPS module connected to Raspberry Pi via Serial.
    Requires: pip install pyserial
    Default port for RPi: /dev/ttyS0 or /dev/ttyAMA0
    """
    try:
        # Update port to match your hardware connection
        # For USB GPS use /dev/ttyUSB0
        ser = serial.Serial('/dev/ttyS0', baudrate=9600, timeout=1)
        
        # We try for a few seconds to get a valid fix
        timeout = time.time() + 5
        while time.time() < timeout:
            line = ser.readline().decode('ascii', errors='replace')
            if line.startswith('$GPRMC'):
                data = line.split(',')
                if data[2] == 'A':  # Status A = valid fix
                    # Convert to decimal degrees
                    lat_raw = float(data[3])
                    lat_deg = int(lat_raw/100)
                    lat_min = lat_raw - (lat_deg * 100)
                    latitude = lat_deg + (lat_min/60)
                    if data[4] == 'S': latitude = -latitude

                    lon_raw = float(data[5])
                    lon_deg = int(lon_raw/100)
                    lon_min = lon_raw - (lon_deg * 100)
                    longitude = lon_deg + (lon_min/60)
                    if data[6] == 'W': longitude = -longitude

                    return latitude, longitude
        ser.close()
    except Exception as e:
        print(f"GPS Hardware Error: {e}")
    
    return None, None

if __name__ == "__main__":
    print("Testing GPS hardware connection...")
    lat, lon = get_hardware_gps()
    if lat:
        print(f"Fix found! Latitude: {lat}, Longitude: {lon}")
    else:
        print("No GPS fix or hardware not connected.")
