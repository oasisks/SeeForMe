import serial
import time

# Set up the serial connection
ser = serial.Serial('COM3', 9600, timeout=1)  # Change to your port (e.g., "/dev/ttyUSB0" for Linux)
time.sleep(2)

if ser.is_open:
    print("Serial port opened successfully!")
else:
    print("Failed to open serial port!")
ser.flush()

ser.write(b'WARN: RIGHT ON\n')
time.sleep(0.5)
ser.write(b'WARN: FORWARD ON\n')
time.sleep(0.5)
ser.write(b'WARN: FORWARD OFF\n')
time.sleep(0.5)
ser.write(b'WARN: LEFT ON\n')
time.sleep(0.5)
ser.write(b'WARN: RIGHT OFF\n')
time.sleep(0.5)
ser.write(b'WARN: LEFT OFF\n')
time.sleep(0.5)

# power = 0

# while True:
#     # Read the line from the serial port (each line is sent by Arduino)
#     if ser.in_waiting > 0:
#         button_state = ser.readline().decode('utf-8').strip()  # Read the serial data
        
#         print(f"Button State: {button_state}")
        
#         # Control motor PWM based on button state
#         if "STRONG" in button_state and not power == 2:
#             power = 2
#             ser.write(b'POWER: 100\n')  # Send PWM value to Arduino to turn the motor at full speed
#         elif "WEAK" in button_state and not power == 1:
#             power = 1
#             ser.write(b'POWER: 50\n')  # Send PWM value to Arduino to turn the motor at lower speed
#         elif "NONE" in button_state and not power == 0:
#             power = 0
#             ser.write(b'POWER: 0\n')
    
#     time.sleep(0.1)  # Delay to avoid constantly querying the serial port
