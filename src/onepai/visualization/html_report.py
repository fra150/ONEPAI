"""
HTMLReport — Report di introspezione autocontenuto in un singolo file HTML.

Tema "Neural Observatory": dark, glassmorphism, accenti cyan/viola con glow,
sfondo a rete neurale animata, contatori animati, sezioni reveal-on-scroll.
Zero dipendenze frontend: tutto vanilla JS + SVG inline + canvas.

Uso::

    report = HTMLReport(title="MyModel Introspection")
    report.add_summary({"layers": 12, "dead_neurons": 3, "ghost_heads": 2})
    report.add_section("Silence Map", silence_heatmap.render(data), icon="signal")
    report.save("onepai_report.html")
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["HTMLReport"]

_CSS = """
:root {
  --bg-0:#04060d; --bg-1:#070b16; --panel:rgba(16,22,40,0.55);
  --panel-solid:#0b1120; --brd:rgba(120,165,255,0.16); --brd-glow:rgba(80,200,255,0.40);
  --cyan:#38e1ff; --violet:#a274ff; --green:#3df5a6; --amber:#ffb74a; --red:#ff5d72;
  --ghost:#8090b0; --text:#dfe8ff; --muted:#8493b8; --mono:ui-monospace,"JetBrains Mono","SF Mono",Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
}
*,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }
html { scroll-behavior:smooth; }
body { font-family:var(--sans); background:var(--bg-0); color:var(--text);
  font-size:14px; line-height:1.55; -webkit-font-smoothing:antialiased; overflow-x:hidden; }
body::before { content:""; position:fixed; inset:0; z-index:-2;
  background:
    radial-gradient(1200px 700px at 78% -8%, rgba(60,120,255,0.10), transparent 60%),
    radial-gradient(900px 600px at 8% 108%, rgba(150,90,255,0.10), transparent 55%),
    var(--bg-0);
  background-attachment:fixed; }
#bg-canvas { position:fixed; inset:0; z-index:-1; opacity:0.55; }
.scanlines { position:fixed; inset:0; z-index:1; pointer-events:none; opacity:0.35;
  background:repeating-linear-gradient(to bottom, rgba(0,0,0,0) 0, rgba(0,0,0,0) 2px, rgba(0,0,0,0.18) 3px); }
a { color:var(--cyan); text-decoration:none; }
.sr-only { position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0,0,0,0); }

/* status bar */
#statusbar { position:fixed; top:0; left:0; right:0; height:36px; z-index:200;
  display:flex; align-items:center; gap:16px; padding:0 18px;
  background:rgba(6,10,20,0.82); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
  border-bottom:1px solid var(--brd); font-family:var(--mono); font-size:11px; letter-spacing:0.08em; }
#statusbar .live-dot { width:8px; height:8px; border-radius:50%; background:var(--green);
  box-shadow:0 0 10px var(--green); animation:pulse 1.8s infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1);} 50%{opacity:0.45;transform:scale(0.7);} }
#statusbar .sb-brand { color:var(--cyan); font-weight:500; text-shadow:0 0 14px rgba(56,225,255,0.6); }
#statusbar .sb-sep { color:var(--muted); }
#statusbar .sb-meta { color:var(--muted); }
#statusbar #clock { margin-left:auto; color:var(--text); }

/* sidebar */
#sidebar { position:fixed; top:36px; left:0; bottom:0; width:240px; z-index:100; overflow-y:auto;
  background:rgba(7,11,22,0.62); backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px);
  border-right:1px solid var(--brd); padding-bottom:30px; }
#sidebar .brand-block { padding:22px 20px 16px; border-bottom:1px solid var(--brd); }
#sidebar .brand-glyph { font-family:var(--mono); font-size:26px; color:var(--cyan);
  text-shadow:0 0 18px rgba(56,225,255,0.7); animation:spin 9s linear infinite; display:inline-block; }
@keyframes spin { to { transform:rotate(360deg); } }
#sidebar .brand-name { font-size:15px; font-weight:500; letter-spacing:0.22em; margin-top:6px; }
#sidebar .brand-sub { font-family:var(--mono); font-size:10px; color:var(--muted); letter-spacing:0.18em; margin-top:3px; }
#sidebar nav { padding:12px 0; }
#sidebar nav a { display:flex; align-items:center; gap:11px; padding:10px 20px; color:var(--muted);
  font-size:12.5px; border-left:2px solid transparent; transition:all .18s; position:relative; }
#sidebar nav a .nav-ix { font-family:var(--mono); font-size:10px; color:var(--brd-glow); min-width:22px; }
#sidebar nav a:hover { color:var(--text); background:rgba(56,225,255,0.05); }
#sidebar nav a.active { color:var(--cyan); background:rgba(56,225,255,0.08);
  border-left-color:var(--cyan); box-shadow:inset 0 0 22px rgba(56,225,255,0.08); }
#sidebar nav a.active .nav-ix { color:var(--cyan); }

/* main */
#main { margin-left:240px; padding:60px 40px 80px; max-width:1180px; position:relative; z-index:2; }
.hero { margin-bottom:30px; }
.hero h1 { font-size:27px; font-weight:500; letter-spacing:0.01em;
  text-shadow:0 0 26px rgba(56,225,255,0.25); }
.hero h1 .accent { color:var(--cyan); }
.hero .boot { font-family:var(--mono); font-size:12px; color:var(--muted); margin-top:9px;
  border-left:2px solid var(--cyan); padding-left:11px; min-height:17px; }
.hero .boot .ok { color:var(--green); }

/* cards */
.cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(176px,1fr)); gap:14px; margin-bottom:30px; }
.card { position:relative; background:var(--panel); border:1px solid var(--brd); border-radius:12px;
  padding:18px 16px 16px; backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); overflow:hidden; }
.card::before { content:""; position:absolute; top:0; left:0; right:0; height:2px;
  background:var(--cyan); box-shadow:0 0 16px var(--cyan); opacity:0.9; }
.card.c-danger::before { background:var(--red); box-shadow:0 0 16px var(--red); }
.card.c-warn::before { background:var(--amber); box-shadow:0 0 16px var(--amber); }
.card.c-ok::before { background:var(--green); box-shadow:0 0 16px var(--green); }
.card.c-violet::before { background:var(--violet); box-shadow:0 0 16px var(--violet); }
.card .card-value { font-family:var(--mono); font-size:30px; font-weight:500; line-height:1;
  color:var(--text); text-shadow:0 0 18px rgba(120,180,255,0.25); }
.card .card-label { font-size:10.5px; color:var(--muted); margin-top:9px;
  text-transform:uppercase; letter-spacing:0.12em; }
.card .card-corner { position:absolute; bottom:8px; right:10px; font-family:var(--mono);
  font-size:9px; color:var(--brd-glow); opacity:0.7; }

/* sections */
.section { background:var(--panel); border:1px solid var(--brd); border-radius:14px;
  margin-bottom:22px; overflow:hidden; backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);
  opacity:0; transform:translateY(18px); transition:opacity .6s ease, transform .6s ease; position:relative; }
.section.visible { opacity:1; transform:translateY(0); }
.section::before, .section::after { content:""; position:absolute; width:14px; height:14px; pointer-events:none; opacity:0.6; }
.section::before { top:8px; left:8px; border-top:1.5px solid var(--brd-glow); border-left:1.5px solid var(--brd-glow); }
.section::after { bottom:8px; right:8px; border-bottom:1.5px solid var(--brd-glow); border-right:1.5px solid var(--brd-glow); }
.section-header { padding:15px 22px; cursor:pointer; display:flex; align-items:center; gap:13px;
  border-bottom:1px solid var(--brd); user-select:none; transition:background .18s; }
.section-header:hover { background:rgba(56,225,255,0.04); }
.section-header .sx { font-family:var(--mono); font-size:11px; color:var(--cyan); letter-spacing:0.1em; }
.section-header .icon { font-size:17px; filter:drop-shadow(0 0 8px rgba(56,225,255,0.5)); }
.section-header h2 { font-size:16px; font-weight:500; flex:1; letter-spacing:0.01em; }
.section-header .toggle { font-family:var(--mono); font-size:11px; color:var(--muted); }
.section-body { padding:22px; overflow-x:auto; }
.section-body.collapsed { display:none; }
.svg-wrap { overflow-x:auto; }
.svg-wrap svg { max-width:100%; height:auto; display:block; }

/* note */
.note { background:rgba(56,225,255,0.06); border-left:2px solid var(--cyan); padding:10px 15px;
  font-size:12.5px; color:var(--text); border-radius:0 7px 7px 0; margin-bottom:15px; line-height:1.6; }

/* tables */
table { width:100%; border-collapse:collapse; font-size:12.5px; font-family:var(--mono); }
th { padding:10px 13px; text-align:left; font-size:10px; text-transform:uppercase; letter-spacing:0.1em;
  color:var(--cyan); border-bottom:1px solid var(--brd-glow); background:rgba(56,225,255,0.04); }
td { padding:9px 13px; border-bottom:1px solid rgba(120,165,255,0.07); color:var(--text); }
tr:hover td { background:rgba(56,225,255,0.04); }
.badge { display:inline-block; padding:2px 9px; border-radius:20px; font-size:10px; font-weight:500;
  font-family:var(--mono); letter-spacing:0.04em; }
.badge-ghost { background:rgba(128,144,176,0.16); color:#aab6d4; border:1px solid rgba(128,144,176,0.35); }
.badge-shadow { background:rgba(255,183,74,0.13); color:var(--amber); border:1px solid rgba(255,183,74,0.4); }
.badge-dead { background:rgba(255,93,114,0.13); color:var(--red); border:1px solid rgba(255,93,114,0.42); }
.badge-active { background:rgba(61,245,166,0.12); color:var(--green); border:1px solid rgba(61,245,166,0.4); }

/* footer */
.foot { margin-top:34px; padding-top:18px; border-top:1px solid var(--brd);
  font-family:var(--mono); font-size:10.5px; color:var(--muted); letter-spacing:0.08em;
  display:flex; gap:14px; flex-wrap:wrap; }
.foot .ok { color:var(--green); }

@media (max-width:820px) {
  #sidebar { display:none; }
  #main { margin-left:0; padding:54px 16px 60px; }
}
"""

_JS = """
(function(){
  var c=document.getElementById('bg-canvas'); if(c){
    var ctx=c.getContext('2d'), N=56, nodes=[], DPR=Math.min(window.devicePixelRatio||1,2);
    function resize(){ c.width=innerWidth*DPR; c.height=innerHeight*DPR; ctx.setTransform(DPR,0,0,DPR,0,0); }
    resize(); addEventListener('resize',resize);
    for(var i=0;i<N;i++){ nodes.push({x:Math.random()*innerWidth,y:Math.random()*innerHeight,vx:(Math.random()-0.5)*0.28,vy:(Math.random()-0.5)*0.28}); }
    function tick(){
      ctx.clearRect(0,0,innerWidth,innerHeight);
      for(var i=0;i<N;i++){ var n=nodes[i]; n.x+=n.vx; n.y+=n.vy;
        if(n.x<0||n.x>innerWidth)n.vx*=-1; if(n.y<0||n.y>innerHeight)n.vy*=-1; }
      for(var i=0;i<N;i++) for(var j=i+1;j<N;j++){
        var a=nodes[i],b=nodes[j],dx=a.x-b.x,dy=a.y-b.y,d=Math.sqrt(dx*dx+dy*dy);
        if(d<150){ ctx.strokeStyle='rgba(56,225,255,'+(0.11*(1-d/150))+')'; ctx.lineWidth=1;
          ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke(); } }
      for(var i=0;i<N;i++){ var n=nodes[i]; ctx.fillStyle='rgba(130,180,255,0.55)';
        ctx.beginPath(); ctx.arc(n.x,n.y,1.5,0,6.2832); ctx.fill(); }
      requestAnimationFrame(tick);
    } tick();
  }
  function easeOut(t){ return 1-Math.pow(1-t,3); }
  document.querySelectorAll('[data-count]').forEach(function(el){
    var target=parseFloat(el.getAttribute('data-count')), dec=parseInt(el.getAttribute('data-dec')||'0',10), t0=null;
    function step(ts){ if(!t0)t0=ts; var p=Math.min((ts-t0)/900,1);
      el.textContent=(target*easeOut(p)).toFixed(dec); if(p<1)requestAnimationFrame(step); }
    requestAnimationFrame(step);
  });
  var io=new IntersectionObserver(function(es){ es.forEach(function(e){
    if(e.isIntersecting){ e.target.classList.add('visible'); io.unobserve(e.target); } }); },{threshold:0.06});
  document.querySelectorAll('.section').forEach(function(s,i){ s.style.transitionDelay=((i%3)*70)+'ms'; io.observe(s); });
  document.querySelectorAll('.section-header').forEach(function(h){
    h.addEventListener('click',function(){ var b=h.nextElementSibling,t=h.querySelector('.toggle');
      b.classList.toggle('collapsed'); if(t)t.textContent=b.classList.contains('collapsed')?'[+]':'[-]'; }); });
  var secs=document.querySelectorAll('.section[id]'), navs=document.querySelectorAll('#sidebar nav a');
  addEventListener('scroll',function(){ var y=scrollY;
    secs.forEach(function(s){ var top=s.offsetTop-90, bot=top+s.offsetHeight;
      if(y>=top&&y<bot) navs.forEach(function(a){ a.classList.toggle('active',a.getAttribute('href')==='#'+s.id); }); }); },{passive:true});
  navs.forEach(function(a){ a.addEventListener('click',function(e){ var t=document.querySelector(a.getAttribute('href'));
    if(t){ e.preventDefault(); t.scrollIntoView({behavior:'smooth',block:'start'}); } }); });
  function clock(){ var el=document.getElementById('clock'); if(el) el.textContent=new Date().toISOString().slice(11,19)+' UTC'; }
  clock(); setInterval(clock,1000);
  var boot=document.getElementById('boot');
  if(boot){ var msg=boot.getAttribute('data-msg')||'', i=0;
    (function type(){ if(i<=msg.length){ boot.innerHTML=msg.slice(0,i)+'<span style="color:#3df5a6">_</span>'; i++; setTimeout(type,16); }
      else boot.innerHTML=msg+' <span class="ok">[ ok ]</span>'; })(); }
})();
"""


def _slug(text: str) -> str:
    return text.lower().replace(" ", "-").replace("/", "-")[:32]


def _is_number(v: Any) -> bool:
    try:
        float(str(v).replace(",", ""))
        return True
    except (ValueError, TypeError):
        return False


def _decimals(v: Any) -> int:
    s = str(v)
    return len(s.split(".")[1]) if "." in s and s.split(".")[1].isdigit() else 0


class HTMLReport:
    """
    Assembla un report HTML "Neural Observatory" con sidebar, card animate
    e sezioni collassabili. Ogni sezione accetta HTML/SVG grezzo.
    """

    def __init__(self, title: str = "ONEPAI Introspection Report",
                 model_name: str = ""):
        self.title = title
        self.model_name = model_name
        self._summary: Dict[str, Any] = {}
        self._sections: List[Dict] = []
        self._tables: List[Dict] = []

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def add_summary(self, metrics: Dict[str, Any]) -> "HTMLReport":
        """Aggiunge card metriche in cima al report."""
        self._summary.update(metrics)
        return self

    def add_section(
        self,
        title: str,
        content: str,
        icon: str = "◉",
        note: Optional[str] = None,
        collapsed: bool = False,
    ) -> "HTMLReport":
        """Aggiunge una sezione con contenuto HTML/SVG (es. SVG da heatmap.render())."""
        self._sections.append({
            "title": title, "icon": icon, "content": content,
            "note": note, "collapsed": collapsed, "id": _slug(title),
        })
        return self

    def add_table(
        self,
        title: str,
        rows: List[Dict[str, Any]],
        icon: str = "≡",
        note: Optional[str] = None,
    ) -> "HTMLReport":
        """Aggiunge una sezione con una tabella (chiavi dei dict = header)."""
        self._tables.append({"title": title, "icon": icon, "rows": rows,
                             "note": note, "id": _slug(title)})
        return self

    def add_from_silence_report(self, silence_report) -> "HTMLReport":
        """Shortcut: cards + tabella da un SilenceReport
        (dead_neurons, silent_layers, analyzed_observations)."""
        dead = getattr(silence_report, "dead_neurons", []) or []
        silent = getattr(silence_report, "silent_layers", []) or []
        all_entities = list(dead) + list(silent)
        self.add_summary({
            "Dead neurons": len(dead),
            "Silent layers": len(silent),
            "Analyzed observations": getattr(silence_report, "analyzed_observations", "?"),
        })
        if all_entities:
            rows = [
                {
                    "Name": e.name,
                    "Type": e.entity_type,
                    "Silence ratio": f"{e.silence_ratio:.3f}",
                    "Observations": e.total_observations,
                    "Status": _badge(e.entity_type),
                }
                for e in all_entities
            ]
            self.add_table("Silent Entities", rows, icon="◍")
        return self

    def add_from_gradient_report(self, report: Dict) -> "HTMLReport":
        """Shortcut: info dal flow_report() di GradientFlowTracer."""
        layers = report.get("layers", {})
        dead = report.get("dead_parameters", [])
        self.add_summary({
            "Tracked layers": len(layers),
            "Dead parameters": len(dead),
            "Health": report.get("health", "?"),
        })
        if dead:
            rows = [{"Layer": ld, "Param": pn, "Mean magnitude": f"{mag:.2e}"}
                    for ld, pn, mag in dead]
            self.add_table("Dead Parameters", rows, icon="▽")
        return self

    def add_from_attention_summary(self, summaries: list) -> "HTMLReport":
        """Shortcut: tabella da AttentionTracer.summarize()."""
        ghost_count = sum(1 for s in summaries if s.ghost_score > 0.8)
        self.add_summary({
            "Attention heads": len(summaries),
            "Ghost heads": ghost_count,
        })
        rows = [
            {
                "Layer": s.layer_id,
                "Head": s.head_idx,
                "Max weight": f"{s.max_weight_ever:.4f}",
                "Entropy": f"{s.mean_entropy:.3f}",
                "Ghost score": f"{s.ghost_score:.2f}",
                "Status": _badge_head(s.ghost_score),
                "Obs": s.observations,
            }
            for s in sorted(summaries, key=lambda s: s.ghost_score, reverse=True)
        ]
        self.add_table("Attention Head Analysis", rows, icon="◉")
        return self

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def to_html(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        all_navs = self._sections + self._tables

        nav_items = "\n".join(
            f'<a href="#{s["id"]}"><span class="nav-ix">{i:02d}</span>'
            f'<span class="icon" aria-hidden="true">{s["icon"]}</span>{s["title"]}</a>'
            for i, s in enumerate(all_navs, 1)
        )

        # card accent per chiave
        def _card_class(key: str) -> str:
            k = key.lower()
            if "dead" in k:
                return "c-danger"
            if "ghost" in k:
                return "c-violet"
            if "silent" in k or "silence" in k:
                return "c-warn"
            if "health" in k or "ok" in k:
                return "c-ok"
            return ""

        cards_html = ""
        for n, (k, v) in enumerate(self._summary.items(), 1):
            if _is_number(v):
                dec = _decimals(v)
                val_html = (f'<span class="card-value" data-count="{float(str(v).replace(",", ""))}" '
                            f'data-dec="{dec}">0</span>')
            else:
                val_html = f'<span class="card-value">{v}</span>'
            cards_html += (
                f'<div class="card {_card_class(k)}">{val_html}'
                f'<div class="card-label">{k}</div>'
                f'<div class="card-corner">#{n:02d}</div></div>'
            )

        # sezioni SVG/HTML
        sections_html = ""
        sec_index = 0
        for sec in self._sections:
            sec_index += 1
            body_class = "section-body collapsed" if sec["collapsed"] else "section-body"
            tog = "[+]" if sec["collapsed"] else "[-]"
            note_html = f'<div class="note">{sec["note"]}</div>' if sec["note"] else ""
            sections_html += f"""
<div class="section" id="{sec["id"]}">
  <div class="section-header">
    <span class="sx">§{sec_index:02d}</span>
    <span class="icon" aria-hidden="true">{sec["icon"]}</span>
    <h2>{sec["title"]}</h2>
    <span class="toggle">{tog}</span>
  </div>
  <div class="{body_class}">
    {note_html}
    <div class="svg-wrap">{sec["content"]}</div>
  </div>
</div>"""

        # sezioni tabella
        for tab in self._tables:
            if not tab["rows"]:
                continue
            sec_index += 1
            headers = list(tab["rows"][0].keys())
            th_row = "".join(f"<th>{h}</th>" for h in headers)
            tbody = "".join(
                "<tr>" + "".join(f"<td>{row.get(h, '')}</td>" for h in headers) + "</tr>"
                for row in tab["rows"]
            )
            note_html = f'<div class="note">{tab["note"]}</div>' if tab["note"] else ""
            sections_html += f"""
<div class="section" id="{tab["id"]}">
  <div class="section-header">
    <span class="sx">§{sec_index:02d}</span>
    <span class="icon" aria-hidden="true">{tab["icon"]}</span>
    <h2>{tab["title"]}</h2>
    <span class="toggle">[-]</span>
  </div>
  <div class="section-body">
    {note_html}
    <table><thead><tr>{th_row}</tr></thead><tbody>{tbody}</tbody></table>
  </div>
</div>"""

        model_tag = self.model_name or "model"
        boot_msg = (f"initializing shadow map :: tracing silence :: "
                    f"target={model_tag} :: {len(all_navs)} modules online")
        cards_block = f'<div class="cards">{cards_html}</div>' if cards_html else ''

        return f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{self.title}</title>
  <style>{_CSS}</style>
</head>
<body>
<canvas id="bg-canvas" aria-hidden="true"></canvas>
<div class="scanlines" aria-hidden="true"></div>
<h2 class="sr-only">{self.title}: report di introspezione del modello {model_tag}.</h2>

<header id="statusbar">
  <span class="live-dot" aria-hidden="true"></span>
  <span class="sb-brand">ONEPAI</span>
  <span class="sb-sep">//</span>
  <span class="sb-meta">neural observatory</span>
  <span class="sb-sep">//</span>
  <span class="sb-meta">target: {model_tag}</span>
  <span id="clock">--:--:-- UTC</span>
</header>

<aside id="sidebar">
  <div class="brand-block">
    <span class="brand-glyph" aria-hidden="true">◈</span>
    <div class="brand-name">ONEPAI</div>
    <div class="brand-sub">INTROSPECTION v0.2</div>
  </div>
  <nav>{nav_items}</nav>
</aside>

<main id="main">
  <div class="hero">
    <h1>{self.title}</h1>
    <div class="boot" id="boot" data-msg="{boot_msg}"></div>
  </div>
  {cards_block}
  {sections_html}
  <div class="foot">
    <span>ONEPAI <span class="ok">●</span> introspection engine</span>
    <span>//</span><span>generated {now}</span>
    <span>//</span><span>{len(all_navs)} modules</span>
  </div>
</main>
<script>{_JS}</script>
</body>
</html>"""

    def save(self, path: str | Path = "onepai_report.html") -> Path:
        """Salva il report su disco e ritorna il percorso assoluto."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_html(), encoding="utf-8")
        return p.resolve()

    # ------------------------------------------------------------------
    # Factory rapida
    # ------------------------------------------------------------------

    @classmethod
    def from_pipeline(
        cls,
        registry=None,
        shadow_map=None,
        silence_report=None,
        attention_tracer=None,
        gradient_tracer=None,
        logit_lens=None,
        title: str = "ONEPAI Pipeline Report",
        model_name: str = "",
    ) -> "HTMLReport":
        """Costruisce un report completo dagli oggetti del pipeline ONEPAI.
        Ogni argomento è opzionale."""
        from .heatmap import ActivationHeatmap, SilenceHeatmap
        from .timeline_viz import ActivationTimeline, GradientTimeline, ConfidenceTimeline

        report = cls(title=title, model_name=model_name)

        if registry is not None:
            hm = ActivationHeatmap(cell_size=20)
            report.add_section("Activation Overview", hm.render_from_registry(registry),
                               icon="◬", note="Media dell'activation_value per layer nel registry.")

        if shadow_map is not None:
            shm = SilenceHeatmap(cell_size=26)
            report.add_section("Silence Map", shm.render_from_shadow_map(shadow_map),
                               icon="◍", note="Verde scuro = neuroni silenziosi (silence_ratio→1).")

        if silence_report is not None:
            report.add_from_silence_report(silence_report)

        if attention_tracer is not None:
            report.add_from_attention_summary(attention_tracer.summarize())
            try:
                inf_mat = attention_tracer.build_influence_matrix(seq_len=16)
                from .heatmap import AttentionHeatmap
                ahm = AttentionHeatmap(cell_size=22)
                report.add_section("Attention Flow Matrix",
                                   ahm.render(inf_mat, title="Aggregate Attention Flow"),
                                   icon="◉", collapsed=True)
            except Exception:
                pass

        if gradient_tracer is not None:
            report.add_from_gradient_report(gradient_tracer.flow_report())
            gt = GradientTimeline()
            trajectories = {lid: gradient_tracer.get_magnitude_trajectory(lid)
                            for lid in list({k[0] for k in gradient_tracer._records})[:8]}
            if trajectories:
                report.add_section("Gradient Timeline",
                                   gt.render(trajectories, title="Gradient Magnitude per Layer"),
                                   icon="▽")

        if logit_lens is not None:
            ct = ConfidenceTimeline()
            report.add_section("Logit Lens — Confidence", ct.render_from_logit_lens(logit_lens),
                               icon="◎", note="Come la confidenza del modello cresce strato dopo strato.")
            s = logit_lens.summary()
            report.add_summary({
                "Layers (lens)": s.get("layers", 0),
                "Positions tracked": s.get("positions_tracked", 0),
                "Mean confidence": f"{s.get('mean_confidence', 0):.3f}",
            })

        return report


# ------------------------------------------------------------------
# Helpers badge HTML
# ------------------------------------------------------------------

def _badge(name: str) -> str:
    n = name.lower()
    if "dead" in n:
        return '<span class="badge badge-dead">dead</span>'
    if "shadow" in n or "ghost" in n:
        return '<span class="badge badge-shadow">shadow</span>'
    if "layer" in n:
        return '<span class="badge badge-shadow">silent</span>'
    return '<span class="badge badge-active">active</span>'


def _badge_head(ghost_score: float) -> str:
    if ghost_score > 0.8:
        return '<span class="badge badge-ghost">ghost</span>'
    if ghost_score > 0.4:
        return '<span class="badge badge-shadow">shadow</span>'
    return '<span class="badge badge-active">active</span>'
