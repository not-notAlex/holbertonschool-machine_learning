#!/usr/bin/env python3
"""
module for task 3
"""

import sys
import requests as rq
import time


if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    r = rq.get(url)
    launches = sorted(r.json(), key=lambda i: i['date_unix'])
    date_unix = launches[0]['date_unix']
    for i in r.json():
        if i['date_unix'] == date_unix:
            launch_name = i['name']
            date = i['date_local']
            rocket_id = i['rocket']
            launchpad_id = i['launchpad']
            break
    r = rq.get("https://api.spacexdata.com/v4/rockets/{}".format(rocket_id))
    rocket_name = r.json()['name']
    r = rq.get(
        "https://api.spacexdata.com/v4/launchpads/{}".format(launchpad_id))
    lpad_name = r.json()['name']
    lpad_locality = r.json()['locality']
    print("{} ({}) {} - {} ({})".format(
        launch_name, date, rocket_name, lpad_name, lpad_locality))
