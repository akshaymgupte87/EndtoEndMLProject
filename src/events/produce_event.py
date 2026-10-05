"""Send one demonstration interaction to Kafka."""

import argparse
import time

from kafka import KafkaProducer

from src.events.schema import encode_event


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brokers", default="localhost:9092")
    parser.add_argument("--topic", default="recommender-events")
    parser.add_argument("--user-idx", type=int, required=True)
    parser.add_argument("--item-idx", type=int, required=True)
    parser.add_argument("--event-type", choices=("click", "view"), default="click")
    args = parser.parse_args()
    payload = encode_event({"user_idx": args.user_idx, "item_idx": args.item_idx, "event_type": args.event_type, "event_time": int(time.time())})
    producer = KafkaProducer(bootstrap_servers=args.brokers, value_serializer=lambda value: value)
    try:
        result = producer.send(args.topic, payload).get(timeout=15)
        print(f"sent topic={result.topic} partition={result.partition} offset={result.offset}")
    finally:
        producer.close()


if __name__ == "__main__":
    main()
