from paho.mqtt import client as mqtt_client
from json import loads,dumps
from kafka import KafkaProducer
from threading import Thread
import datetime
import time

broker = "mqtt.eclipseprojects.io"
broker_port = 1883
topic = "Movie"
client_name = "aClient"


def on_connect(client,userdata,msg,rc):
    if rc==0 :
        print("Connected to MQTT broker")
    else:
        print(f"Connection failed : {rc}")


def on_message(client,userdata,msg):
    print(f"Received Topic: {msg.topic} Value: {msg.payload.decode()}")
    pkt = msg.payload.decode()
    kafkasend(pkt)



def initProducer():
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer = lambda x: dumps(x).encode('utf-8'),
        api_version=(0, 10, 1)
    )
    return producer

def kafkasend(pkt):
    producer = initProducer()
    #pkt['received_timestamp'] = str(datetime.datetime.now())
    kafka_topic = "basua_topic" #str(pkt['topic']).replace('/','_')
    print(f"Sent kafka topic {kafka_topic}")
    kafka_value = pkt
    producer.send(kafka_topic, value=kafka_value,partition=0)
    time.sleep(1)


def start_mqtt_client():
    mqttc = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION1,client_name)
    mqttc.connect(broker,broker_port)
    mqttc.on_connect = on_connect
    mqttc.on_message =on_message
    mqttc.loop_start()
    mqttc.subscribe(topic)
    time.sleep(3)
    mqttc.loop_stop()

if __name__ == "__main__" :
    # Start the MQTT client in a separate thread
    mqtt_thread = Thread(target=start_mqtt_client)
    mqtt_thread.start()

    

