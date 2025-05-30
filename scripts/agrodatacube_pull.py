import os
import requests
import urllib.parse


def main():

    url = "https://agrodatacube.wur.nl/api/v2/rest/fields"

    geometry = "POLYGON((219478 481588.256000001,219478 497638.432,237339.072000001 497638.432,237339.072000001 481588.256000001,219478 481588.256000001))"
    encoded_geometry = urllib.parse.quote(geometry)

    params = {
        "geometry": encoded_geometry,
        "fieldid": "9403114",
        "output_epsg": "4326",
        "page_size": "25",
        "page_offset": "0",
        "year": "2018",
        "cropcode": "265",
        "epsg": "28992",
        "noclip": ""
    }
    headers = {
        "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3N1ZWR0byI6InlrZS52YW5yYW5kZW5Ad3VyLm5sIiwicmVzb3VyY2UiOlsiKiJdLCJyZXF1ZXN0X2xpbWl0Ijo1MDAsImFyZWFfbGltaXQiOjEwMDAwMDAuMCwiZXhwIjoxNTQ4ODg5MjAwLCJpYXQiOjE1MjUyNTg5OTksImlzc3VlZGRhdGUiOjE1MjUyNTg5OTl9.dy6ixBrgBdI1R8xNbA4qlRSnK6Rv12w8hfY4PlrNQ1k"
        'Accept: application/json'
    }

    response = requests.get(url, params=params, headers=headers)
    print(response.status_code)
    print(response.text)

if __name__ == '__main__':
    main()