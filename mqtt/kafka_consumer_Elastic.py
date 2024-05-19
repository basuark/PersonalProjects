
import json
import logging
import signal
from kafka import KafkaConsumer
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ElasticsearchException

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# Function to create and configure Kafka consumer
def create_kafka_consumer(topic_name, bootstrap_servers, group_id):
    consumer = KafkaConsumer(
        topic_name,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset='earliest',
        enable_auto_commit=False,
        group_id=group_id,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    return consumer

# Function to create and configure Elasticsearch client
def create_elasticsearch_client(hosts):
    es = Elasticsearch(hosts)
    return es

# Main function to consume and index messages
def consume_and_index(kafka_topic, kafka_servers, es_hosts, es_index):
    consumer = create_kafka_consumer(kafka_topic, kafka_servers, 'my-group')
    es = create_elasticsearch_client(es_hosts)

    for message in consumer:
        try:
            # Index the document in Elasticsearch
            res = es.index(index=es_index, body=message.value)
            #logging.info(f"Indexed document ID: {res['_id']} to Elasticsearch")
            consumer.commit()
        except ElasticsearchException as e:
            #logging.error(f"Elasticsearch indexing failed: {e}")
            # Handle exception (e.g., retry, log, or raise)
            pass
    # Close the consumer when done
    consumer.close()

if __name__ == '__main__':
    KAFKA_TOPIC = 'your_kafka_topic'
    KAFKA_SERVERS = ['localhost:9092']
    ES_HOSTS = ['http://localhost:9200']
    ES_INDEX = 'your_elasticsearch_index'

    consume_and_index(KAFKA_TOPIC, KAFKA_SERVERS, ES_HOSTS, ES_INDEX)

    