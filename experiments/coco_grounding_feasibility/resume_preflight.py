"""Run only inside the existing failed Kaggle notebook; reuse prepared HDF5, zero prior updates."""
import base64, hashlib, json, shutil, sys, zlib
from pathlib import Path

assert 'REPO' in globals() and 'OUTPUT' in globals() and callable(globals().get('run')), 'Run inside the notebook that failed preflight, in the same live session.'
PROTOCOL=Path(OUTPUT)/'protocol.json'
cfg=json.loads(PROTOCOL.read_text())
assert Path(cfg['stage']).is_file(), 'Existing staged HDF5 missing; do not restart training blindly.'
assert not (Path(OUTPUT)/'checkpoint.bin').exists(), 'A checkpoint already exists; preserve completed work.'
assert not list(Path(OUTPUT).glob('history_rank*.json')), 'Training may have started; do not apply a preflight-only recovery.'
name='routing_preflight.py'
old=cfg['implementation_sha256'][name]
assert old in ('9914f95e02e994c2ebdf7a31f3314641c4185bcf2dd64cb8e3f0c6277f5492fa', '1e1d87f75d705a79eabd63218f44c1df181f2ae442d03279b03c14e1fa62f06e'), 'Unexpected preflight version; preserve source/protocol.'
current=Path(REPO)/name
assert hashlib.sha256(current.read_bytes()).hexdigest()==old, 'Runtime source no longer matches protocol.'
source=zlib.decompress(base64.b64decode('eNqlWetv2zgS/56/gk0/iGoZNck2e3sJdEBv0+4Wl7ZBmrsDzjAIWqJtrmVRFaU8WvR/v5kh9bDjpIe9oEhtcjgvzuM3zP7+/lVbNmatWbbU2coxUzqTa6ZVtmSVKWzDStvombWrM/ZV15bZCsgNfGRtlatGu2R/f39vXts1w6/Ey6wrWzcMP+e6aNReWPjD2bL7bJ0/VKlmWZhZd+YSvnYkja2zzS9JblxTm1nb6Jwpx/CrZ+O3yzKpVK2KQhcdw/PhxLlq1GW3DafPzy+DDmioBPVVd8qpG+33agtHZVXbWW/XFS5d4orw24KV7VrXJpPei4JdXX+6EOwN/B64mHIh1zYfVFsUdqYK6SpVyrVWpWCqvTOFUfX93t5erucsz6vAkteqXInCZqqIT/cY/GSFco5d23ve2/7B5m2hwz7+IA8pTWkaKbnTxXy0hz+urXTN46Snic+QKqnSnic6bK0bIPNLjS6drflxInJ9YzKdep3iDZlzW9+qOieR4kYVrXZbkmvdtHX5wAXci3/hzySuXfNYFLrkgUk8OZwSn8bep3B/HO2PgyrS5C6dkD5Trw/owfxJoe8qnWHYmJJNOJ8cJYL+TcXkL8k0FsfJSSxgHQyjf7BOq7Tovxwm8XSwAjRIMCPkolZoaSMbK0tb6vS6bvXgj8I6B+683/Sf12qCtzrd8uQZnkhmKluRD+ORRM/AYSglcPu6bmQG1JqjMmu6/aRKUCGxIa0zflOSqBtbpIdC0X9eDoYnMKPPzyG6VrpkeDHs1jRLcF2z1AxvS+Cnkqlwg7TmCTkUDDB6weycqGlxDdrGCXHFhHchvtB/jp+IY3H01583lBO1/tKaWjty78ilDjZ1umHdJHktkl+23Dj5CFchTgX+Nx0Jpt8vicsrEEo7M9VkyzQ3WcMpEuFyTL4lY4IqTLeECCLPFNREW24f+OkBdX+TdKyyzuA5t31wciiOprtFNXghu/UjBYXX8mnBcBmrTui8LQrOj8QR3kAskpNdUqFIuIf2vVOF04+LMmu1wJTcOnZ0PN1VOaBsN85fweiegDKDilnLL6h0WKSqcHAUi6xwEld27L04HAqAqtdirjJQw+d+9PcI8h7yOrqKwOBxTmPmCZn2ZZiTWoLiQwCfIRe7jEoP+I9j8YUX//JIH/wSJ5AbUHKp2v1PuU1K9RlMOUvthdIWW0wcugV1Ctlg7waXwQkFzdyW3M6A2w2cXeu1re9FpfIcpIhG1QsN5lE8hPps5tjwgzqQgqroj0+icCCadkdjBk59nJ4YA/VYABV/ZZxm/8IS+LauwWvRtVeahVRiFOfuVdAULFPlQudJFHdKjoSQUdE0cUtVaegPz1JsGH45fnGCKg7UgePj5E9p+bbrIfoOLrS4Z3Nzo6GVzXWty0w7Boni477TFOOPFjD0ajSCj6SNXIKEnfE96cl207S3KXF7cfIyEG/sP3l5nZ8mwGUaQmFC3KY/uMbeZ/5o+Nqd3VTx6ev1UplxJI0YHKzVHyAdPXnQe5L1wYuO9NENGEzPC7NYQoWobWMz2yEhQljBH5g9LnONrAOuDVgrs+XcLNpaS4SDWb8vGEEymZsF5B7gOb8MJaU0c1gZBMCO9Ii4Z1lXrRNZmyvpK4TTOpcALzcPBezXHSusgoomZq0pcg/k5osUkXGCO44jAh4sTGqtcsjpu4YDMgBOKe3DmUkE36q2iQLWGQBiasqGW5fo8sbUtpxEV28+/gPIxPb6xadf31zIsNtl1jbRvz9dXZzLz+//8xaInqXHpw+v95/w/fr13TFFMfRbLLldBviQQicliJF8XeS+Lp75TVwHAC2bJRrr+HF89th18QEIgKoj3ouetywBrwYBz1IIPVco0C7aofeFzVaQzUuAWQC19CkZ0SmOs0VCsBguA5IbkYhtKx6VWVZEArXB6+iHHL42JcwCLj0+DD3NI/d0sjkchH4RHBFlVRvFsXiaBiyMAviPPZjZPRl4hBSikL8+DpbAYAN4cJX6iMXgiQfcAxr7kMR10dGKqAF/lZEg7oHRc3aOkwAYCr4xIBBKYKNWmjW31s8SN+DinF1fvXn/EauVO4NJal0VkPqugemwy/xat0iH4UJiMGuDy6DjlelkOlTPzcoZVE5QT4ibcQmFgNjanZjpJBoQHRR9VcK4cOqlJKqqdJlzE49ZoBC/H6cQ7DMIydWQz/5eIJ8LLySR/jOkakH2+RyHeqXaAhq4Xx6BzKFa8C0iPnmo/eABr9I0RBbVrJRKCN0aXFh8NtNArdNxPeP0xZ/xmAKiI6wmdCOSIFcMtn6LZiA1iV6uSOoKpXpCWh9TfyeGOBGMZmE+EMc4XdPIPNzOJvbwWiZ+0V/qCAt5cJHS8iQKlR2ARDIHRwHQ4gCQB3JEF1u0BHEfo+8aW/rt+8bQimeh4nA/RwHcWziBI9hie3btGCT+BYR3+iIpAIrQW/3Xo4B+Us8JMNSikSt9L4MjgqbiQRMd/3RO63h4ATvYDEYCYMqLEAxJQNHQSxaQuIANw3yOLVUurV3xYLvAO5VezNYY29T3m14AGGvAaC8hgE6+G2FOTsXpQfBE+DwwnpsSC8mp1xh0XNsbPULGfxLVPikzv00pPJNbjYACXBpcFE3Psgd7X8Bda9zrz4dcym97FJlugU4whmWPbPdsnrNf0TgGYBIQFws6QOJBW3cdyEzYuZkTLmrYb28/fPCFhDnzFQBnCbUeK+qIJSiftwCh1kiGuOPd5U/H4baoHOsDzyE4sMbZyzSEysjXwHGmC3ub7A3xh8MZPXXx/FaAXbWutGokYAVdF1rBhZ2Iw1iELCz0QmUDJIdbAE+V+lYCNnCc6nh+G9PEOXo4gqnr0dmL8A6ginGIA+X2Kwm+FUDD9U8gVMaqoYxV3UuW4zFW+kAVWtLwThBQNPHqgDG205m1RWjMxs0RGWi+iBPcimMStkBhdK5H1Nhw5tBfG75ISggjIP3b4TbxUzPHR0RCKOsVPb/27qGjBoKiwyveyB+9SFHRDo97gTD+U0W6KqBv/3/Z/8TMS9yFD1ooXw3w5/Hu2bef2zMN4UMKeV3p6Wf06zDBPglhkEuT3/1QBWDXP4LO2zLDvqKKJKtBlgS/17a693putJmR1UenU0gUygF6rzALiAANSZPru3SsytOGbQc5xOeOHv8s9f1/B8y97Aanbo5mobR1oYOv3Ryy+9U86ocsicDvG/76nuB0Egn/PgYwoHVpdHn19t3F+99+v5aXbz5/9jAxJazY/4FAhj8QpIej96f7sllqwI5peClHHNuW+kaXEkqmpPDNLIB933/EGhCSubXQq/DBRIYC5fdGLeRGy354BOwLnl7DFW2S963DL6cRFdjdxffswXA6GvOjQbKPBgo36aMVtF9jQcUHM3ohE1hyN0l8wX3EGDBg174vApjO5COoA7KrAG6LAyQ2ZCAMAUPF8xQew0FAHp/8HMJFhMEq3R58eVeaaQ7KYaW291ujUHi6q3FkHCKCYUQImkfD+BDt/stRJOZF65ahOvkp/9EhfwN/7HoL2BuDiTGi94Ocoz9xGIiNrxpr3pNm/Rd5XR8X')).decode()
new=hashlib.sha256(source.encode()).hexdigest()
assert new=='1e1d87f75d705a79eabd63218f44c1df181f2ae442d03279b03c14e1fa62f06e'
backup=PROTOCOL.with_name('protocol_before_preflight_fix.json')
if not backup.exists():shutil.copyfile(PROTOCOL,backup)
current.write_bytes(source.encode('utf-8'))
assert hashlib.sha256(current.read_bytes()).hexdigest()==new
cfg['implementation_sha256'][name]=new
PROTOCOL.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
if isinstance(globals().get('SOURCES'),dict):SOURCES[name]=source
if isinstance(globals().get('SOURCE_HASHES'),dict):SOURCE_HASHES[name]=new
receipt=dict(reason='Replace cross-batch FP32 logits comparison with exact actual five-reference decoder-input checks; retain same-batch parity and original tolerance.',
    old_preflight_sha256=old,new_preflight_sha256=new,model_data_loss_recipe_changed=False,staging_reused=True)
(Path(OUTPUT)/'preflight_fix_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
try:
    for mode in ('preflight','worker'):
        run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0',
             'run_routing.py',mode,'--config',PROTOCOL],cwd=REPO)
    run([sys.executable,'-u','run_routing.py','evaluate','--config',PROTOCOL],cwd=REPO)
finally:
    if Path(LOG).is_file():shutil.copyfile(LOG,Path(OUTPUT)/'notebook.log')
    saved=Path(OUTPUT)/'source'/name;saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(source.encode('utf-8'))
    run([sys.executable,'-u','run_routing.py','archive','--config',PROTOCOL],cwd=REPO)
print('SEND REVIEW:',str(OUTPUT)+'_review.zip',flush=True)
