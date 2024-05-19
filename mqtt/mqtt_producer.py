from paho.mqtt import client as mqttc

import time

broker = "mqtt.eclipseprojects.io"
broker_port = 1883
topic = "Movie"
client_name = 'ibasuABCD'

doc = {"name" : "Avatar 2", "Type" : "Aliens" }

mqtt_client = mqttc.Client(mqttc.CallbackAPIVersion.VERSION1,client_name)
mqtt_client.connect(broker)

while True:
    mqtt_client.publish(topic=topic, payload=str(doc))
    print(f"Sent topic: {topic}")
    time.sleep(5)
