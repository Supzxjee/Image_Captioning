"""Use diagram-design exporters; use installed Chromium rather than download a browser."""
from pathlib import Path
import importlib.util,json,subprocess
from playwright.sync_api import sync_playwright,BrowserType,Page
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from functools import partial
from threading import Thread
from urllib.parse import urlparse,quote
ROOT=Path(__file__).resolve().parent
SKILL=Path('/home/agent/.codex/plugins/cache/diagram-design/diagram-design/2.6.142/skills/diagram-design')
spec=importlib.util.spec_from_file_location('diagram_png',SKILL/'scripts/export_png.py')
export=importlib.util.module_from_spec(spec);spec.loader.exec_module(export)
original=BrowserType.launch

def system_chromium(self,*a,**kw):
 kw.setdefault('executable_path','/usr/bin/chromium')
 return original(self,*a,**kw)
BrowserType.launch=system_chromium
# The managed browser serves local artifacts over HTTP rather than file://.
class QuietHandler(SimpleHTTPRequestHandler):
 def log_message(self,*args): pass
server=ThreadingHTTPServer(('127.0.0.1',0),partial(QuietHandler,directory=str(ROOT)))
Thread(target=server.serve_forever,daemon=True).start()
original_goto=Page.goto
def local_goto(self,url,*a,**kw):
 parsed=urlparse(url)
 if parsed.scheme=='file':
  url=f'http://127.0.0.1:{server.server_port}/'+quote(Path(parsed.path).name)+(('?'+parsed.query) if parsed.query else '')
 return original_goto(self,url,*a,**kw)
Page.goto=local_goto
reports={}
for html in sorted(ROOT.glob('*.html')):
 subprocess.run(['python',str(SKILL/'scripts/export_svg.py'),str(html),'--system-fonts'],check=True,capture_output=True)
 export.rasterize(html,html.with_suffix('.png'),2)
 print('PNG',html.stem,flush=True)
with sync_playwright() as p:
 browser=p.chromium.launch()
 for html in sorted(ROOT.glob('*.html')):
  page=browser.new_page()
  page.goto(html.as_uri(),wait_until='load')
  page.evaluate('document.fonts.ready')
  audit=page.evaluate('''() => {
   const svg=document.querySelector('svg'), v=svg.viewBox.baseVal;
   const nodes=new Map([...svg.querySelectorAll('[data-node-id]')].map(n=>[n.dataset.nodeId,n.getBBox()]));
   const outside=[];
   for(const t of svg.querySelectorAll('text[data-owner]')){
    let b=t.getBBox(), n=nodes.get(t.dataset.owner);
    if(b.x<n.x+5 || b.x+b.width>n.x+n.width-5 || b.y<n.y+3 || b.y+b.height>n.y+n.height-3)
     outside.push({node:t.dataset.owner,text:t.textContent,bbox:{x:b.x,y:b.y,w:b.width,h:b.height},nodebox:{x:n.x,y:n.y,w:n.width,h:n.height}});
   }
   const dims=[...svg.querySelectorAll('text')].filter(t=>t.textContent==='150×512');
   const collisions=[];
   for(const id of ['noun-aggregation','verb-aggregation']) {
    const g=svg.querySelector('g[data-icon-kind][data-module="'+id+'"]');
    const a=g.getBoundingClientRect();
    for(const t of dims){const b=t.getBoundingClientRect();
     if(a.left<b.right && a.right>b.left && a.top<b.bottom && a.bottom>b.top) collisions.push({icon:id,text:t.textContent});
    }
   }
   return {aggregationBadgeLabelOverlap:collisions,viewBox:[v.x,v.y,v.width,v.height],overflow:outside,textNodes:svg.querySelectorAll('text').length,font:getComputedStyle(svg.querySelector('text')).fontFamily};
  }''')
  w,h=audit['viewBox'][2:]
  page.pdf(path=str(html.with_suffix('.pdf')),width=f'{w}px',height=f'{h}px',print_background=True,margin=dict(top='0',right='0',bottom='0',left='0'))
  reports[html.stem]=audit
  print('PDF/audit',html.stem,'overflow',len(audit['overflow']),flush=True)
  page.close()
 browser.close()
(ROOT/'render_audit.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2))

server.shutdown()
