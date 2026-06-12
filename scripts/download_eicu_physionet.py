#!/usr/bin/env python
"""Resumable downloader for the PhysioNet eICU-CRD file directory.

The script prompts for PhysioNet credentials locally, lists the credentialed
file directory, and downloads files with HTTP Range resume support.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import html.parser
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from http.cookiejar import MozillaCookieJar


BASE_URL = "https://physionet.org/files/eicu-crd/2.0/"


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


def build_opener(username: str, password: str, proxy: str | None) -> urllib.request.OpenerDirector:
    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(None, BASE_URL, username, password)
    handlers: list[urllib.request.BaseHandler] = [
        urllib.request.HTTPBasicAuthHandler(password_mgr),
        urllib.request.HTTPDigestAuthHandler(password_mgr),
    ]
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    elif proxy == "":
        handlers.append(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(*handlers)


def build_cookie_opener(cookie_file: Path, proxy: str | None) -> urllib.request.OpenerDirector:
    jar = MozillaCookieJar(str(cookie_file))
    jar.load(ignore_discard=True, ignore_expires=True)
    handlers: list[urllib.request.BaseHandler] = [urllib.request.HTTPCookieProcessor(jar)]
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    elif proxy == "":
        handlers.append(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(*handlers)


def request_with_auth(
    opener: urllib.request.OpenerDirector,
    url: str,
    username: str | None,
    password: str | None,
    headers: dict[str, str] | None = None,
) -> urllib.request.addinfourl:
    req_headers = {
        "User-Agent": "eicu-resume-downloader/1.0",
    }
    if username is not None and password is not None:
        req_headers["Authorization"] = "Basic " + base64.b64encode(
            f"{username}:{password}".encode("utf-8")
        ).decode("ascii")
    if headers:
        req_headers.update(headers)
    return opener.open(urllib.request.Request(url, headers=req_headers), timeout=60)


def list_files(
    opener: urllib.request.OpenerDirector,
    username: str | None,
    password: str | None,
) -> list[str]:
    try:
        with request_with_auth(opener, BASE_URL, username, password) as response:
            html = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise SystemExit(
                "PhysioNet returned HTTP 403 while listing eICU files.\n"
                "Most common causes:\n"
                "  1. The username/password is not accepted for command-line downloads.\n"
                "  2. Your account is credentialed for MIMIC but has not signed the eICU-CRD DUA.\n"
                "  3. eICU-CRD access is still pending.\n\n"
                "Check https://physionet.org/content/eicu-crd/2.0/ in your browser while logged in. "
                "If the page shows Request Access or Sign Data Use Agreement, complete that first.\n"
                "If the browser can download files, try a one-file command to bypass directory listing:\n"
                "  python scripts\\download_eicu_physionet.py --out D:\\eicu-crd-2.0 "
                "--proxy http://127.0.0.1:10798 --only patient.csv.gz"
            ) from exc
        raise

    parser = LinkParser()
    parser.feed(html)
    files: list[str] = []
    for href in parser.links:
        if href.startswith("?") or href.startswith("#") or href.startswith("../"):
            continue
        if href.endswith("/"):
            continue
        name = Path(urllib.parse.urlparse(href).path).name
        if name:
            files.append(name)
    return sorted(set(files))


def remote_size(
    opener: urllib.request.OpenerDirector,
    url: str,
    username: str | None,
    password: str | None,
) -> int | None:
    try:
        with request_with_auth(opener, url, username, password, {"Range": "bytes=0-0"}) as response:
            content_range = response.headers.get("Content-Range")
            if content_range and "/" in content_range:
                return int(content_range.rsplit("/", 1)[1])
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def download_file(
    opener: urllib.request.OpenerDirector,
    name: str,
    out_dir: Path,
    username: str | None,
    password: str | None,
    retries: int,
) -> None:
    out_path = out_dir / name
    part_path = out_dir / f"{name}.part"
    url = urllib.parse.urljoin(BASE_URL, name)
    expected = remote_size(opener, url, username, password)

    if expected is not None and out_path.exists() and out_path.stat().st_size == expected:
        print(f"[skip] {name} already complete ({expected / 1024 / 1024:.1f} MiB)")
        return

    start = part_path.stat().st_size if part_path.exists() else 0
    if expected is not None and start > expected:
        part_path.unlink()
        start = 0

    for attempt in range(1, retries + 1):
        headers = {}
        mode = "ab"
        if start:
            headers["Range"] = f"bytes={start}-"
        else:
            mode = "wb"

        try:
            with request_with_auth(opener, url, username, password, headers) as response:
                status = getattr(response, "status", response.getcode())
                if start and status == 200:
                    print(f"[restart] server ignored resume for {name}; starting from zero")
                    mode = "wb"
                    start = 0

                total_label = f"{expected / 1024 / 1024:.1f} MiB" if expected else "unknown size"
                print(f"[download] {name} from {start / 1024 / 1024:.1f} MiB / {total_label}")

                written = start
                tick_bytes = written
                tick_time = time.time()
                with part_path.open(mode + "b" if "b" not in mode else mode) as fh:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
                        written += len(chunk)
                        now = time.time()
                        if now - tick_time >= 10:
                            speed = (written - tick_bytes) / (now - tick_time) / 1024 / 1024
                            if expected:
                                pct = written / expected * 100
                                print(f"  {pct:5.1f}%  {written / 1024 / 1024:.1f} MiB  {speed:.2f} MiB/s")
                            else:
                                print(f"  {written / 1024 / 1024:.1f} MiB  {speed:.2f} MiB/s")
                            tick_bytes = written
                            tick_time = now

            if expected is None or part_path.stat().st_size == expected:
                part_path.replace(out_path)
                print(f"[done] {name}")
                return

            start = part_path.stat().st_size
            print(f"[retry] incomplete {name}: {start} of {expected} bytes")
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                raise SystemExit(
                    "PhysioNet refused access. Check that your account has eICU-CRD access "
                    "and that the Data Use Agreement is signed."
                ) from exc
            print(f"[retry] HTTP {exc.code} for {name} on attempt {attempt}/{retries}")
        except Exception as exc:
            print(f"[retry] {name} failed on attempt {attempt}/{retries}: {exc}")

        time.sleep(min(60, attempt * 5))
        start = part_path.stat().st_size if part_path.exists() else 0

    raise RuntimeError(f"failed to download {name} after {retries} attempts")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download eICU-CRD 2.0 files from PhysioNet.")
    parser.add_argument("--out", default=r"D:\eicu-crd-2.0", help="output directory")
    parser.add_argument("--user", default=os.environ.get("PHYSIONET_USER"), help="PhysioNet username")
    parser.add_argument(
        "--cookies",
        type=Path,
        help="Netscape/Mozilla cookie file exported from a logged-in PhysioNet browser session",
    )
    parser.add_argument("--proxy", default=None, help="proxy URL, for example http://127.0.0.1:10798")
    parser.add_argument("--no-proxy", action="store_true", help="ignore system/browser proxy")
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument("--only", nargs="*", help="download only these filenames")
    args = parser.parse_args()

    proxy = "" if args.no_proxy else args.proxy
    if args.cookies:
        username = None
        password = None
        opener = build_cookie_opener(args.cookies, proxy)
    else:
        username = args.user or input("PhysioNet username: ").strip()
        password = getpass.getpass("PhysioNet password: ")
        opener = build_opener(username, password, proxy)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.only) if args.only else list_files(opener, username, password)
    if not files:
        raise SystemExit("No files found. Check access permissions or the PhysioNet directory listing.")

    print(f"Found {len(files)} files. Output: {out_dir}")
    for name in files:
        download_file(opener, name, out_dir, username, password, args.retries)

    print("All requested files are complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
