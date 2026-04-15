import requests
from google.transit import gtfs_realtime_pb2

url = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace"

response = requests.get(url)

feed = gtfs_realtime_pb2.FeedMessage()
feed.ParseFromString(response.content)

for entity in feed.entity:
    if entity.HasField("trip_update"):
        print(entity.trip_update)
        break