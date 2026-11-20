"""Fetch aligned PiLLVD frames (ll_final, nl_final) for chosen clips with HTTP range requests.
Data: https://zenodo.org/records/23015904 (CC BY 4.0, credit Abu Awwad, Rosin, Karakus, arXiv:2611.16515)."""
import json, struct, subprocess, zlib, os, sys, hashlib
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__))
BASE="https://zenodo.org/api/records/23015904/files/%s/content"
idx=json.load(open(os.path.join(HERE,"zip_index.json")))
CLIPS=[("Outdoor.zip","Outdoor/person04","outdoor_person04"),("Indoor-P4.zip","person/person01","indoor_person01"),
("Outdoor.zip","Outdoor/person05","outdoor_person05"),("Outdoor.zip","Outdoor/person02","outdoor_person02"),
("Indoor-P1.zip","drawing","drawing"),("Indoor-P2.zip","hand-gesture/hand-gesture02","hand_gesture02"),
("Indoor-P3.zip","whiteboard","whiteboard")]  # existing files are skipped
OUT=os.path.join(HERE,"..","data")
def rng(url,a,b):
    for t in range(4):
        r=subprocess.run(["curl","-sL","--fail","-r",f"{a}-{b}",url],capture_output=True)
        if r.returncode==0 and len(r.stdout)==b-a+1: return r.stdout
    raise RuntimeError(f"range failed {a}-{b}")
def fetch(job):
    z,nm,csz,off,dest=job
    if os.path.exists(dest): return 0
    url=BASE%z
    h=rng(url,off,off+29); nl,el=struct.unpack("<HH",h[26:30]); meth=struct.unpack("<H",h[8:10])[0]
    s=off+30+nl+el; data=rng(url,s,s+csz-1)
    if meth==8: data=zlib.decompress(data,-15)
    if data[:2]!=b"\xff\xd8": raise RuntimeError("not a jpeg: "+nm)
    open(dest+".part","wb").write(data); os.rename(dest+".part",dest); return len(data)
jobs=[]
for z,clip,name in CLIPS:
    for kind in("ll_final","nl_final"):
        os.makedirs(f"{OUT}/{name}/{kind}",exist_ok=True)
        for nm,csz,usz,off in idx[z]:
            if nm.startswith(f"{clip}/{kind}/") and nm.endswith(".jpg"):
                jobs.append((z,nm,csz,off,f"{OUT}/{name}/{kind}/{os.path.basename(nm)}"))
print(len(jobs),"files",flush=True)
done=0;tot=0
with ThreadPoolExecutor(4) as ex:
    for n in ex.map(fetch,jobs):
        done+=1;tot+=n
        if done%100==0: print(done,round(tot/1e6),"MB",flush=True)
print("DONE",done,round(tot/1e6),"MB",flush=True)
