import serial
import time

# Set up the serial connection
ser = serial.Serial('COM3', 9600, timeout=1)  # Change COM3 to your port (e.g., "/dev/ttyUSB0" for Linux)
time.sleep(2)

def set_motor_speed(speed):
    """Send motor speed (0-255) to Arduino."""
    if 0 <= speed <= 255:
        ser.write(f"{speed}\n".encode())  # Send speed as string
    else:
        print("Speed must be between 0 and 255.")

# Example usage
set_motor_speed(150)  # Set motor to medium speed
time.sleep(2)
set_motor_speed(0)  # Turn off motor

ser.close()