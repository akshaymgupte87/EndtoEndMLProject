"""Read a bounded number of Kafka events and write JSON Lines for a future batch."""

import argparse
import json
from pathlib import Path

from kafka import KafkaConsumer

from src.events.schema import decode_event


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brokers", default="localhost:9092")
    parser.add_argument("--topic", default="recommender-events")
    parser.add_argument("--group", default="recommender-event-demo")
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("data/events/events.jsonl"))
    args = parser.parse_args()
    if args.max_events < 1:
        parser.error("--max-events must be positive")
    consumer = KafkaConsumer(args.topic, bootstrap_servers=args.brokers, group_id=args.group, auto_offset_reset="earliest", enable_auto_commit=False, consumer_timeout_ms=10_000)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with args.output.open("a", encoding="utf-8") as stream:
            for record in consumer:
                stream.write(json.dumps(decode_event(record.value), separators=(",", ":")) + "\n")
                count += 1
                if count >= args.max_events:
                    break
            stream.flush()
            if count:
                consumer.commit()
    finally:
        consumer.close()
    print(f"wrote {count} events to {args.output}")


if __name__ == "__main__":
    main()
