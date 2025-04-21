#define LEFT 9
#define FORWARD 10
#define RIGHT 11

void setup() {
  pinMode(LEFT, OUTPUT);
  pinMode(FORWARD, OUTPUT);
  pinMode(RIGHT, OUTPUT);
  Serial.begin(9600);  // Start serial communication at 9600 baud
}

void loop() {
  // Read the serial input from the Python script
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');  // reads until newline character
    if (input.startsWith("WARN: ")) {
      input = input.substring(6);  // get everything after the first 5 characters
      int dir = 0;

      if (input.startsWith("LEFT")) {
        dir = LEFT;
      } else if (input.startsWith("FORWARD")) {
        dir = FORWARD;
      } else if (input.startsWith("RIGHT")) {
        dir = RIGHT;
      } else {
        return;
      }

      if (input.endsWith("ON")) {
        digitalWrite(dir, HIGH);
      } else if (input.endsWith("OFF")) {
        digitalWrite(dir, LOW);
      } else {
        return;
      } 
    }
  }
}
