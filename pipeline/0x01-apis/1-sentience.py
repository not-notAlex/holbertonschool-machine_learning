#!/usr/bin/env python3
"""
module for task 1
"""

import requests as rq


def sentientPlanets():
    """
    returns names of planets of all sentient species
    """
    planets = []
    page = 1
    while True:
        url = "https://swapi-api.hbtn.io/api/species/?page=" + str(page)
        r = rq.get(url)
        data = r.json()
        results = data['results']
        for specie in results:
            if specie['classification'] == 'sentient' or specie[
                    'designation'] == 'sentient':
                homeworld = specie['homeworld']
                if homeworld is not None:
                    req = rq.get(specie['homeworld'])
                    Data = req.json()
                    planets.append(Data['name'])
        if data['next'] is None:
            break
        page += 1
    return planets
