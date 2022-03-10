#!/usr/bin/env python3
"""
module for task 0
"""

import requests as rq


def availableShips(passengerCount):
    """
    returns a list of ships that can hold a given number of passengers
    """
    vehicles = []
    page = 1
    while True:
        url = "https://swapi-api.hbtn.io/api/starships/?page=" + str(page)
        r = rq.get(url)
        data = r.json()
        results = data['results']
        for vehicle in results:
            passenger = vehicle['passengers']
            passenger = passenger.replace(',', "")
            if passenger.isnumeric() and int(passenger) >= passengerCount:
                vehicles.append(vehicle['name'])
        page += 1
        if data['next'] is None:
            break
    return vehicles
