from typing import Dict, List, Set

import numpy as np
import pymongo
import pymongo.database


def event_stats(mongo_client: pymongo.MongoClient) -> List[Dict]:
    db: pymongo.database.Database = mongo_client["opq"]
    box_events_coll: pymongo.collection.Collection = db["box_events"]
    events_coll: pymongo.collection.Collection = db["events"]

    events_query = {"target_event_start_timestamp_ms": {"$gt": 0},
                    "target_event_end_timestamp_ms": {"$gt": 0}}

    events_projection = {"_id": False,
                         "target_event_start_timestamp_ms": True,
                         "target_event_end_timestamp_ms": True}

    events: List[Dict] = events_coll.find(events_query, projection=events_projection)
    durations_events_ms: np.ndarray = np.array(
        list(map(lambda doc: doc["target_event_end_timestamp_ms"] - doc["target_event_start_timestamp_ms"], events)))
    durations_events_s: np.ndarray = durations_events_ms / 1_000.0

    data_duration_events_s = durations_events_s.sum()

    box_events_query = {"event_start_timestamp_ms": {"$gt": 0},
                        "event_end_timestamp_ms": {"$gt": 0}}
    box_events_projection = {"_id": False,
                             "event_id": True,
                             "event_start_timestamp_ms": True,
                             "event_end_timestamp_ms": True}

    box_events: pymongo.cursor.Cursor = box_events_coll.find(box_events_query, projection=box_events_projection)
    box_events: List[Dict] = list(box_events)

    min_ts_ms: int = min(list(map(lambda doc: doc["event_start_timestamp_ms"], box_events)))
    max_ts_ms: int = max(list(map(lambda doc: doc["event_end_timestamp_ms"], box_events)))

    durations_box_events_ms: np.ndarray = np.array(
        list(map(lambda doc: doc["event_end_timestamp_ms"] - doc["event_start_timestamp_ms"], box_events)))
    durations_box_events_s: np.ndarray = durations_box_events_ms / 1000.0
    data_stored: np.ndarray = 12_000.0 * 2 * durations_box_events_s

    total_duration_ms: int = max_ts_ms - min_ts_ms
    total_duration_s: float = total_duration_ms / 1000.0

    data_duration_box_events_s: float = durations_box_events_s.sum()
    mean_data_duration_box_events_s: float = durations_box_events_s.mean()
    std_data_duration_box_events_s: float = durations_box_events_s.std()

    total_data = data_stored.sum()
    mean_data = data_stored.mean()
    std_data = data_stored.std()
    data_per_second = total_data / total_duration_s
    events_per_second = len(box_events) / total_duration_s

    print(f"total events {len(box_events)}")
    print(f"min ts ms {min_ts_ms} max ts ms {max_ts_ms}")
    print(f"total duration s {total_duration_s}")
    print(f"total data duration box events s {data_duration_box_events_s}")
    print(f"percent data duration box events {data_duration_box_events_s / total_duration_s}")
    print(f"mean data duration box events s, std {mean_data_duration_box_events_s}, {std_data_duration_box_events_s}")
    print(f"percent data duration events {data_duration_events_s / total_duration_s}")
    print(f"total data {total_data}")
    print(f"mean data, std {mean_data}, {std_data}")
    print(f"mean data per second {data_per_second}")
    print(f"mean events per second {events_per_second}")

    return box_events


def incident_stats(mongo_client: pymongo.MongoClient) -> List[Dict]:
    db: pymongo.database.Database = mongo_client["opq"]
    coll: pymongo.collection.Collection = db["incidents"]

    query = {"start_timestamp_ms": {"$gt": 0},
             "end_timestamp_ms": {"$gt": 0},
             "event_id": {"$gt": 0}}
    projection = {"_id": False,
                  "start_timestamp_ms": True,
                  "end_timestamp_ms": True,
                  "classifications": True,
                  "event_id": True}

    incidents: pymongo.cursor.Cursor = coll.find(query, projection=projection)
    incidents: List[Dict] = list(incidents)

    min_ts_ms: int = min(list(map(lambda doc: doc["start_timestamp_ms"], incidents)))
    max_ts_ms: int = max(list(map(lambda doc: doc["end_timestamp_ms"], incidents)))

    durations_ms: np.ndarray = np.array(
        list(map(lambda doc: doc["end_timestamp_ms"] - doc["start_timestamp_ms"], incidents)))
    durations_s: np.ndarray = durations_ms / 1000.0
    data_stored: np.ndarray = 12_000.0 * 2 * durations_s

    total_duration_ms: int = max_ts_ms - min_ts_ms
    total_duration_s: float = total_duration_ms / 1000.0

    data_duration_s: float = durations_s.sum()
    mean_data_duration_s: float = durations_s.mean()
    std_data_duration_s: float = durations_s.std()

    total_data = data_stored.sum()
    mean_data = data_stored.mean()
    std_data = data_stored.std()
    data_per_second = total_data / total_duration_s
    incidents_per_second = len(incidents) / total_duration_s

    print(f"total incidents {len(incidents)}")
    print(f"min ts ms {min_ts_ms} max ts ms {max_ts_ms}")
    print(f"total duration s {total_duration_s}")
    print(f"total data duration s {data_duration_s}")
    print(f"percent data duration {data_duration_s / total_duration_s}")
    print(f"mean data duration s, std {mean_data_duration_s}, {std_data_duration_s}")
    print(f"total data {total_data}")
    print(f"mean data, std {mean_data}, {std_data}")
    print(f"mean data per second {data_per_second}")
    print(f"mean incidents per second {incidents_per_second}")

    return incidents


def phenomena_stats(mongo_client: pymongo.MongoClient):
    import bson
    db: pymongo.database.Database = mongo_client["opq"]
    coll: pymongo.collection.Collection = db["phenomena"]

    query: Dict = {}
    projection: Dict[str, bool] = {}

    phenomena_docs: List[Dict] = list(coll.find(query))
    durations_ms: np.ndarray = np.array(list(map(lambda doc: doc["end_ts_ms"] - doc["start_ts_ms"], phenomena_docs)))
    durations_s: np.ndarray = durations_ms / 1_000.0
    min_ts_ms: float = min(list(map(lambda doc: doc["start_ts_ms"], phenomena_docs)))
    max_ts_ms: float = max(list(map(lambda doc: doc["end_ts_ms"], phenomena_docs)))
    total_duration_ms: float = max_ts_ms - min_ts_ms
    total_duration_s: float = total_duration_ms / 1_000.0
    size_bytes: np.ndarray = np.array(list(map(lambda doc: len(bson.BSON.encode(doc)), phenomena_docs)))
    total_bytes = size_bytes.sum()
    dr_s: float = total_bytes / total_duration_s

    all_incident_ids: Set[int] = set()
    for phenomena_doc in phenomena_docs:
        all_incident_ids.update(phenomena_doc["related_incident_ids"])

    total_incidents = db["incidents"].count()


    print(f"Total Phenomena: {len(phenomena_docs)}")
    print(f"Total bytes: {total_bytes}")
    print(f"Phenomena/s: {len(phenomena_docs) / float(total_duration_s)}")
    print(f"mean size per phenomena: {float(total_bytes) / len(phenomena_docs)}")
    print(f"DR/s: {dr_s}")
    print(f"total_incidents from phenomena: {len(all_incident_ids)}")
    print(f"percent_incident_to_phenomena: {len(all_incident_ids) / float(total_incidents)}")


def ttl_aml_stats(events: List[Dict], incidents: List[Dict]) -> List[Dict]:
    incident_event_ids: Set[str] = set(map(lambda doc: doc["event_id"], incidents))
    events_without_an_incident: List[Dict] = list(filter(lambda doc: doc["event_id"] in incident_event_ids, events))

    min_ts_ms: int = min(list(map(lambda doc: doc["event_start_timestamp_ms"], events)))
    max_ts_ms: int = max(list(map(lambda doc: doc["event_end_timestamp_ms"], events)))

    durations_ms: np.ndarray = np.array(
        list(map(lambda doc: doc["event_end_timestamp_ms"] - doc["event_start_timestamp_ms"],
                 events_without_an_incident)))
    durations_s: np.ndarray = durations_ms / 1000.0

    total_duration_ms: int = max_ts_ms - min_ts_ms
    total_duration_s: float = total_duration_ms / 1000.0
    data_duration_s: float = durations_s.sum()
    seconds_per_week: float = 604800.0
    seconds_per_month: float = seconds_per_week * 4
    data_dur_per_month = (data_duration_s * seconds_per_month) / total_duration_s

    print(f"aml mean data duration per month {data_dur_per_month}")

    events_with_an_incident = len(events) - len(events_without_an_incident)
    print(f"aml percent event to incident {float(events_with_an_incident) / len(events)}")


if __name__ == "__main__":
    mongo_client = pymongo.MongoClient()
    # events = event_stats(mongo_client)
    # print()
    # incidents = incident_stats(mongo_client)
    # print()
    # ttl_aml_stats(events, incidents)
    phenomena_stats(mongo_client)
