#!/usr/bin/env python3
"""Download SCPS OSF cohorts via the OSF API (recursive osfstorage walk). No GPU.
Records into raw/scps/<cond>/osf_<id>/. Logs to download_osf.log."""
import json, os, sys, time, urllib.request as U

DEST = "/projects/EEG-foundation-model/datalake/raw/scps"
LOG = open(os.path.join(DEST, "download_osf.log"), "a")
NODES = [("9827w", "PD"), ("n2abf", "AD"), ("r6w4b", "DEP"), ("pnvay", "SCZ"),
         ("rsnu4", "SCZ"), ("ufet7", "SCZ"), ("zjws3", "ADHD"), ("xzqrv", "ADHD"),
         ("q3u7j", "ASD"), ("26rhz", "ASD"), ("w39ug", "ASD"), ("c2svz", "ASD")]


def log(m):
    s = f"{time.strftime('%H:%M:%S')} {m}"
    print(s); LOG.write(s + "\n"); LOG.flush()


def api(url):
    for _ in range(3):
        try:
            req = U.Request(url, headers={"User-Agent": "scps-dl"})
            return json.load(U.urlopen(req, timeout=40))
        except Exception as e:
            log(f"  api-retry {e}"); time.sleep(3)
    return None


def walk(url, dest):
    """recurse an osfstorage folder listing url, downloading files."""
    while url:
        j = api(url)
        if not j:
            return
        for it in j.get("data", []):
            a = it["attributes"]; kind = a.get("kind")
            name = a.get("name", "f")
            if kind == "file":
                dl = it["links"].get("download"); fp = os.path.join(dest, name)
                if dl and not (os.path.exists(fp) and os.path.getsize(fp) == (a.get("size") or -1)):
                    os.makedirs(dest, exist_ok=True)
                    try:
                        log(f"    get {name} ({(a.get('size') or 0)/1e6:.1f}MB)")
                        U.urlretrieve(dl, fp)
                    except Exception as e:
                        log(f"    FAIL {name} {e}")
            elif kind == "folder":
                sub = it["relationships"]["files"]["links"]["related"]["href"]
                walk(sub, os.path.join(dest, name))
        url = j.get("links", {}).get("next")


def main():
    log(f"START OSF download ({len(NODES)} nodes)")
    for nid, cond in NODES:
        out = os.path.join(DEST, cond, f"osf_{nid}")
        log(f">>> osf {nid} ({cond}) -> {out}")
        walk(f"https://api.osf.io/v2/nodes/{nid}/files/osfstorage/?page[size]=100", out)
        sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out) for f in fs) if os.path.isdir(out) else 0
        log(f"DONE osf {nid} ({sz/1e9:.2f}GB)")
    log("ALL OSF DONE")


if __name__ == "__main__":
    main()
