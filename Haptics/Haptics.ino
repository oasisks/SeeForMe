const int BUTTON_STRONG = 13;
const int BUTTON_WEAK = 12;
const int HAPTIC = 10;

void setup() {
  pinMode(BUTTON_STRONG, INPUT_PULLUP);  // Set the button pin as input with internal pull-up
  pinMode(BUTTON_WEAK, INPUT_PULLUP);  // Set the button pin as input with internal pull-up
  pinMode(HAPTIC, OUTPUT);
  Serial.begin(9600);  // Start serial communication at 9600 baud
}

void loop() {
  // Read the serial input from the Python script
  if (Serial.available() > 0) {
    int pwmValue = Serial.parseInt();  // Read the PWM value sent by Python (from 0 to 255)

    if (pwmValue >= 1 && pwmValue <= 255) {
      analogWrite(HAPTIC, pwmValue);  // Apply PWM to control motor speed
      Serial.print("Motor PWM set to: ");
      Serial.println(pwmValue);
    }
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
