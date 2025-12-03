# ryu_client.py
import requests
import json

class RyuClient:
    def __init__(self, base='http://127.0.0.1:8080'):
        self.base = base.rstrip('/')

    def get_stats(self):
        try:
            r = requests.get(self.base + '/stats', timeout=3.0)
            return r.json()
        except Exception as e:
            print("ryu get_stats error:", e)
            return None

    def set_link_weights(self, deltas):
        payload = {"deltas": deltas}
        try:
            r = requests.post(self.base + '/set_link_weights', json=payload, timeout=3.0)
            return r.status_code == 200
        except Exception as e:
            print("ryu set_link_weights error:", e)
            return False

    def restore_baseline(self):
        try:
            r = requests.post(self.base + '/restore_baseline', timeout=3.0)
            return r.status_code == 200
        except Exception as e:
            print("ryu restore baseline error:", e)
            return False
