import requests
import time
import csv
from google.transit import gtfs_realtime_pb2

# Your feed URL (A/C/E)
FEED_URL = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace"

# Create CSV file
file_name = "realtime_data.csv"

# Write header once
with open(file_name, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp", "trip_id", "route_id", "stop_id", "arrival_time"])

print("🚇 Starting data collection...")

while True:
    try:
        response = requests.get(FEED_URL)

        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(response.content)

        current_time = int(time.time())

        rows = []

        for entity in feed.entity:
            if entity.HasField("trip_update"):
                trip = entity.trip_update

                trip_id = trip.trip.trip_id
                route_id = trip.trip.route_id

                for stop_time in trip.stop_time_update:
                    stop_id = stop_time.stop_id

                    if stop_time.HasField("arrival"):
                        arrival_time = stop_time.arrival.time

                        rows.append([
                            current_time,
                            trip_id,
                            route_id,
                            stop_id,
                            arrival_time
                        ])

        # append to CSV
        with open(file_name, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(rows)

        print(f"Saved {len(rows)} records at {current_time}")

        # wait 30 seconds
        time.sleep(30)

    except Exception as e:
        print("Error:", e)
        time.sleep(10)