#include <Stepper.h>

#define STEPS 2048

Stepper stepper(STEPS, 8, 10, 9, 11);
int serial_data;

void setup()
{
	Serial.begin(9600);
  stepper.setSpeed(12);
}

void loop()
{
    while(Serial.available())
    {
    	serial_data = Serial.read();
    }

    if(serial_data == '1')
    {
    	stepper.step(STEPS);
    }
    else if(serial_data == '0')
    {
    	stepper.step(-STEPS);
    }
}