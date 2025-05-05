const int BUTTON_STRONG = 2;
const int BUTTON_WEAK = 3;
const int HAPTIC = 9;
const int LIGHT = 8;

void setup() {
  pinMode(BUTTON_STRONG, INPUT_PULLUP);  // Set the button pin as input with internal pull-up
  pinMode(BUTTON_WEAK, INPUT_PULLUP);  // Set the button pin as input with internal pull-up
  pinMode(HAPTIC, OUTPUT);
  pinMode(LIGHT, OUTPUT);
  Serial.begin(9600);  // Start serial communication at 9600 baud
}

void loop() {
  // Read the serial input from the Python script
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');  // reads until newline character
    if (input.startsWith("POWER: ")) {
      String numberPart = input.substring(7);  // get everything after the first 5 characters
      int pwmValue = numberPart.toInt();          // convert to integer
      if (pwmValue == 100) {
        digitalWrite(HAPTIC, HIGH);
      } else if (pwmValue == 0) {
        digitalWrite(HAPTIC, LOW);
      }
    }
    // int pwmValue = Serial.parseInt();  // Read the PWM value sent by Python (from 0 to 255)
    // Serial.print("pwmValue: ");
    // Serial.println(pwmValue);
    // if (pwmValue > 0 && pwmValue <= 255) {
    //   // analogWrite(HAPTIC, pwmValue);  // Apply PWM to control motor speed
    //   digitalWrite(HAPTIC, HIGH);
    //   Serial.print("Motor PWM set to: ");
    //   Serial.println(pwmValue);
    // } else if (pwmValue == 0) {
    //   digitalWrite(HAPTIC, LOW);
    // }

  }

  int strongState = digitalRead(BUTTON_STRONG);  // Read the button state
    int weakState = digitalRead(BUTTON_WEAK);  // Read the button state
  if (strongState == LOW) {  // Button is pressed (LOW because using pull-up resistor)
    Serial.println("STRONG");
  } else if (weakState == LOW) {  // Button is pressed (LOW because using pull-up resistor)
    Serial.println("WEAK");
  } else if (strongState == HIGH && weakState == HIGH) {
    Serial.println("NONE");
  }
  delay(100);  // Small delay to avoid flooding the serial
}
