#define LEFT 9
#define FORWARD 11
#define RIGHT 10

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
        input = input.substring(5);
      } else if (input.startsWith("FORWARD")) {
        dir = FORWARD;
        input = input.substring(8);
      } else if (input.startsWith("RIGHT")) {
        dir = RIGHT;
        input = input.substring(6);
      } else {
        return;
      }

      int pwmValue = input.toInt();  // Skips "TEST " and converts "500"
      Serial.print("pwmValue: ");
      Serial.println(pwmValue);
      if (pwmValue >= 0 && pwmValue <= 255) {
        analogWrite(dir, pwmValue);  // Apply PWM to control motor speed
        Serial.print("Motor PWM set to: ");
        // Serial.println(pwmValue);
      }

      // if (input.endsWith("ON")) {
      //   digitalWrite(dir, HIGH);
      // } else if (input.endsWith("OFF")) {
      //   digitalWrite(dir, LOW);
      // } else {
      //   return;
      // } 
    }
  }
}
