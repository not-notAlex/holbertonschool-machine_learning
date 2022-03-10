#!/usr/bin/env python3
"""
module for task 0
"""

import sys
import requests as rq
import time


if __name__ == '__main__':
    r = rq.get("https://api.spacexdata.com/v4/launches")
    launches = {}
    for i in r.json():
        if i['rocket'] not in launches:
            launches[i['rocket']] = 1
        else:
            launches[i['rocket']] += 1
    r = rq.get("https://api.spacexdata.com/v4/rockets/")
    rockets = []
    for i in r.json():
        if i['id'] in launches:
            rockets.append({'rocket': i['name'], 'launches': launches[i['id']]})
    launches = sorted(rockets, key=lambda i: i['rocket'])
    launches = sorted(rockets, key=lambda i: i['launches'], reverse=True)
    for i in launches:
        print("{}: {}".format(i['rocket'], i['launches']))
