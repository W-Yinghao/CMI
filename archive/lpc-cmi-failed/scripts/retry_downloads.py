#!/usr/bin/env python3
"""Retry the failed SCPS downloads with correct host-specific methods:
  - datadryad: file-level public download (dataset-level /download needs auth)
  - figshare:  correct article id 5231053 (MAMEM)
  - OSF:       recurse into child components for nodes whose root osfstorage was ~empty
No GPU. Logs to download_retry.log."""
import json, os, time, urllib.request as U

DEST = "/projects/EEG-foundation-model/datalake/raw/scps"
LOG = open(os.path.join(DEST, "download_retry.log"), "a")


def log(m):
    s = f"{time.strftime('%H:%M:%S')} {m}"; print(s); LOG.write(s + "\n"); LOG.flush()


def api(url):
    for _ in range(3):
        try:
            return json.load(U.urlopen(U.Request(url, headers={"User-Agent": "scps-dl"}), timeout=40))
        except Exception as e:
            log(f"   api-retry {e}"); time.sleep(3)
    return None


def fetch(url, fp):
    if os.path.exists(fp) and os.path.getsize(fp) > 1000:
        log(f"   have {os.path.basename(fp)}"); return
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    try:
        log(f"   GET {os.path.basename(fp)}"); U.urlretrieve(url, fp)
        log(f"      -> {os.path.getsize(fp)/1e6:.1f} MB")
    except Exception as e:
        log(f"   FAIL {os.path.basename(fp)} {e}")


def dryad(doi, cond, name):
    out = os.path.join(DEST, cond, name)
    log(f">>> dryad {doi} ({cond})")
    enc = doi.replace("/", "%2F").replace(":", "%3A")
    ds = api(f"https://datadryad.org/api/v2/datasets/{enc}")
    ver = ds["_links"]["stash:version"]["href"] if ds else None
    if not ver:
        log("   no version"); return
    files = api(f"https://datadryad.org{ver}/files")
    if not files:
        log("   no files listing"); return
    for f in files.get("_embedded", {}).get("stash:files", []):
        nm = f["path"]; dl = f["_links"]["stash:download"]["href"]
        fetch("https://datadryad.org" + dl, os.path.join(out, nm))
    log(f"DONE dryad {doi}")


def figshare(aid, cond, name):
    out = os.path.join(DEST, cond, name)
    log(f">>> figshare {aid} ({cond})")
    j = api(f"https://api.figshare.com/v2/articles/{aid}/files")
    if not j:
        log("   no files"); return
    for f in j:
        fetch(f["download_url"], os.path.join(out, f["name"]))
    log(f"DONE figshare {aid}")


def osf_walk(url, dest):
    while url:
        j = api(url)
        if not j:
            return
        for it in j.get("data", []):
            a = it["attributes"]; nm = a.get("name", "f")
            if a.get("kind") == "file":
                dl = it["links"].get("download")
                if dl:
                    fetch(dl, os.path.join(dest, nm))
            elif a.get("kind") == "folder":
                osf_walk(it["relationships"]["files"]["links"]["related"]["href"], os.path.join(dest, nm))
        url = j.get("links", {}).get("next")


def osf_children(nid, cond):
    out = os.path.join(DEST, cond, f"osf_{nid}")
    log(f">>> osf children of {nid} ({cond})")
    kids = api(f"https://api.osf.io/v2/nodes/{nid}/children/?page[size]=50")
    cids = [c["id"] for c in (kids.get("data", []) if kids else [])]
    log(f"   {len(cids)} child components: {cids}")
    for cid in cids:
        osf_walk(f"https://api.osf.io/v2/nodes/{cid}/files/osfstorage/?page[size]=100", os.path.join(out, cid))
    log(f"DONE osf children {nid}")


def main():
    log("=== RETRY START ===")
    dryad("doi:10.5061/dryad.9cnp5hqk7", "PD", "dryad_PD_pictureorder")
    dryad("doi:10.5061/dryad.8gtht76pw", "DEP", "dryad_DEP_SCZ_padic")
    figshare(5231053, "PD", "figshare_MAMEM")
    osf_children("r6w4b", "DEP")
    osf_children("pnvay", "SCZ")
    log("=== RETRY DONE ===")


if __name__ == "__main__":
    main()
