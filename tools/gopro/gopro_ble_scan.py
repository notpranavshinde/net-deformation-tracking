r"""
Scan for GoPro BLE advertisements without using the interactive SDK demo.

Run:

    python .\tools\gopro\gopro_ble_scan.py
"""

import argparse
import asyncio
import time

from bleak import BleakScanner
from open_gopro.models import AdvData, GoProAdvData


async def scan(seconds: float):
    found = {}
    started = time.time()
    print(f"[INFO] Scanning for GoPros over BLE for {seconds:g}s...")
    async with BleakScanner(service_uuids=["0000fea6-0000-1000-8000-00805f9b34fb"]) as scanner:
        async for device, data in scanner.advertisement_data():
            adv = AdvData()
            adv.update(data)
            if not adv.local_name:
                continue
            try:
                parsed = GoProAdvData.fromAdvData(adv)
            except Exception:
                continue
            serial = str(parsed.serial_number)
            if serial not in found:
                found[serial] = {
                    "serial": serial,
                    "identifier": serial[-4:],
                    "name": adv.local_name,
                    "address": device.address,
                    "rssi": data.rssi,
                }
                print(
                    f"[FOUND] name={adv.local_name} serial={serial} "
                    f"identifier={serial[-4:]} address={device.address} rssi={data.rssi}"
                )
            if time.time() - started >= seconds:
                break
    if not found:
        print("[WARN] No GoPros found. Make sure the cameras are powered on and close to this computer.")
    return found


def main():
    parser = argparse.ArgumentParser(description="Scan GoPro BLE advertisements.")
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()
    asyncio.run(scan(args.seconds))


if __name__ == "__main__":
    main()
