#include <DHT.h>

#define DHTPIN 10
#define DHTTYPE DHT22

DHT dht(DHTPIN, DHTTYPE);

const int gasPin = A0;
const int soilPin = D4;
const int relayPin = D2;

int dryThreshold = 3890;  // ~30% moisture (calibrated: dry=4095, wet=3414)
int gasWarningThreshold = 400;
float heatThreshold = 35.0;

void setup() {
  Serial.begin(9600);
  delay(3000);

  pinMode(relayPin, OUTPUT);
  digitalWrite(relayPin, HIGH); // relay OFF

  dht.begin();
}

void loop() {
  float temp = dht.readTemperature();
  float humidity = dht.readHumidity();

  int gasValue = analogRead(gasPin);
  int soilValue = analogRead(soilPin);

  String status = "STABLE";
  String pumpStatus = "OFF";

  if (isnan(temp) || isnan(humidity)) {
    status = "DHT_ERROR";
  }

  if (soilValue > dryThreshold) {
    digitalWrite(relayPin, LOW);   // pump ON
    pumpStatus = "ON";
    status = "DRY_SOIL";
  } else {
    digitalWrite(relayPin, HIGH);  // pump OFF
  }

  if (!isnan(temp) && temp >= heatThreshold) {
    status = "HEAT_RISK";
  }

  if (gasValue >= gasWarningThreshold) {
    status = "AIR_WARNING";
  }

  Serial.print("soil=");
  Serial.print(soilValue);

  Serial.print(",temp=");
  Serial.print(temp);

  Serial.print(",humidity=");
  Serial.print(humidity);

  Serial.print(",gas=");
  Serial.print(gasValue);

  Serial.print(",status=");
  Serial.print(status);

  Serial.print(",pump=");
  Serial.println(pumpStatus);

  delay(2000);}