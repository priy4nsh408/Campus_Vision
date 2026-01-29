import requests
import os

# API key from environment (.env)
GOOGLE_MAPS_API_KEY = os.environ.get("MAPS_API_KEY", "")

# Test coordinates
AIML_BT = (12.922782, 77.498674)
KRIYA_KALPA = (12.92386, 77.499432)
ADMIN_BLOCK = (12.923885, 77.498632)

directions_url = "https://maps.googleapis.com/maps/api/directions/json"

print("\n=== TEST 1: Route WITHOUT waypoint (should use RVU road) ===")
params_no_waypoint = {
    'origin': f"{AIML_BT[0]},{AIML_BT[1]}",
    'destination': f"{KRIYA_KALPA[0]},{KRIYA_KALPA[1]}",
    'mode': 'walking',
    'key': GOOGLE_MAPS_API_KEY
}

response1 = requests.get(directions_url, params=params_no_waypoint)
data1 = response1.json()

if data1['status'] == 'OK':
    route1 = data1['routes'][0]
    print(f"Distance: {route1['legs'][0]['distance']['text']}")
    print(f"Duration: {route1['legs'][0]['duration']['text']}")
    print("Steps:")
    for i, step in enumerate(route1['legs'][0]['steps'], 1):
        print(f"  {i}. {step['html_instructions'].replace('<b>', '').replace('</b>', '')}")
else:
    print(f"Error: {data1.get('error_message', 'Unknown error')}")

print("\n=== TEST 2: Route WITH 'via:' waypoint (should force through Admin Block) ===")
params_with_via = {
    'origin': f"{AIML_BT[0]},{AIML_BT[1]}",
    'destination': f"{KRIYA_KALPA[0]},{KRIYA_KALPA[1]}",
    'waypoints': f"via:{ADMIN_BLOCK[0]},{ADMIN_BLOCK[1]}",
    'mode': 'walking',
    'key': GOOGLE_MAPS_API_KEY
}

response2 = requests.get(directions_url, params=params_with_via)
data2 = response2.json()

if data2['status'] == 'OK':
    route2 = data2['routes'][0]
    total_dist = 0
    total_dur = 0
    
    print(f"Total Legs: {len(route2['legs'])}")
    for leg_idx, leg in enumerate(route2['legs'], 1):
        print(f"\nLeg {leg_idx}:")
        print(f"  Distance: {leg['distance']['text']}")
        print(f"  Duration: {leg['duration']['text']}")
        total_dist += leg['distance']['value']
        total_dur += leg['duration']['value']
        
    print(f"\nTotal Distance: {total_dist/1000:.2f} km")
    print(f"Total Duration: {total_dur//60} mins")
    
    print("\nAll Steps:")
    for leg_idx, leg in enumerate(route2['legs'], 1):
        for i, step in enumerate(leg['steps'], 1):
            print(f"  Leg{leg_idx}.{i}. {step['html_instructions'].replace('<b>', '').replace('</b>', '')}")
else:
    print(f"Error: {data2.get('error_message', 'Unknown error')}")

print("\n=== TEST 3: Route WITH regular waypoint (NOT via:, should create stop) ===")
params_with_stop = {
    'origin': f"{AIML_BT[0]},{AIML_BT[1]}",
    'destination': f"{KRIYA_KALPA[0]},{KRIYA_KALPA[1]}",
    'waypoints': f"{ADMIN_BLOCK[0]},{ADMIN_BLOCK[1]}",
    'mode': 'walking',
    'key': GOOGLE_MAPS_API_KEY
}

response3 = requests.get(directions_url, params=params_with_stop)
data3 = response3.json()

if data3['status'] == 'OK':
    route3 = data3['routes'][0]
    total_dist = 0
    total_dur = 0
    
    print(f"Total Legs: {len(route3['legs'])}")
    for leg_idx, leg in enumerate(route3['legs'], 1):
        print(f"\nLeg {leg_idx}:")
        print(f"  Distance: {leg['distance']['text']}")
        print(f"  Duration: {leg['duration']['text']}")
        total_dist += leg['distance']['value']
        total_dur += leg['duration']['value']
        
    print(f"\nTotal Distance: {total_dist/1000:.2f} km")
    print(f"Total Duration: {total_dur//60} mins")
else:
    print(f"Error: {data3.get('error_message', 'Unknown error')}")
