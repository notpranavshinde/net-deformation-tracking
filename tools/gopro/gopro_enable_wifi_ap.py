r"""
Enable a GoPro Wi-Fi AP over BLE without asking the SDK to join the network.

Run:

    python .\tools\gopro\gopro_enable_wifi_ap.py --identifier 6714
"""

import argparse
import asyncio
import sys

from open_gopro import WirelessGoPro
from open_gopro.network.wifi import SsidState, WifiController


class NoWifiController(WifiController):
    """Dummy Wi-Fi controller so WirelessGoPro construction stays BLE-only."""

    async def connect(self, ssid: str, password: str, timeout: float = 15) -> bool:
        return False

    async def disconnect(self) -> bool:
        return True

    def current(self):
        return None, SsidState.DISCONNECTED

    def available_interfaces(self):
        return []

    def power(self, power: bool) -> bool:
        return True

    @property
    def is_on(self) -> bool:
        return True


async def enable(identifier: str):
    print(f"[INFO] Preparing BLE-only GoPro connection for identifier={identifier}...", flush=True)
    gopro = WirelessGoPro(
        identifier,
        interfaces={WirelessGoPro.Interface.BLE},
        maintain_state=False,
        wifi_adapter=NoWifiController,
    )
    try:
        print(f"[INFO] Connecting BLE only to GoPro identifier={identifier}...", flush=True)
        gopro._loop = asyncio.get_running_loop()
        gopro._ble_disconnect_event = asyncio.Event()
        await gopro._open_ble(timeout=15, retries=4)

        print("[INFO] Reading camera Wi-Fi credentials over BLE...", flush=True)
        ssid = (await gopro.ble_command.get_wifi_ssid()).data
        password = (await gopro.ble_command.get_wifi_password()).data
        print(f"[INFO] Enabling Wi-Fi AP: SSID={ssid}", flush=True)
        resp = await gopro.ble_command.enable_wifi_ap(enable=True)
        if not resp.ok:
            raise RuntimeError(f"enable_wifi_ap failed: {resp}")

        for _ in range(50):
            status = await gopro.ble_status.ap_mode.get_value()
            if status.ok and status.data:
                print("[OK] GoPro Wi-Fi AP is enabled.", flush=True)
                print(f"SSID: {ssid}", flush=True)
                print(f"Password: {password}", flush=True)
                return
            await asyncio.sleep(0.2)
        raise TimeoutError("Timed out waiting for AP mode status")
    finally:
        await gopro._close_ble()


def main():
    parser = argparse.ArgumentParser(description="Enable GoPro Wi-Fi AP using BLE only.")
    parser.add_argument("--identifier", required=True, help="Last four digits of GoPro serial / default SSID")
    args = parser.parse_args()
    asyncio.run(enable(args.identifier))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
