#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Debug logging, image snapshots, and a tiny web dashboard."""

import functools
import json
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .utils import ensure_dir


class DebugReporter:
    def __init__(self, config, enabled: bool = False, port: Optional[int] = None, host: Optional[str] = None):
        self.config = config
        self.enabled = enabled
        self.root = None
        self.events_file = None
        self.last_image_s = 0.0
        self.httpd = None
        self.thread = None
        self.recent_events = []
        self.host = host or config["debug"].get("host", "127.0.0.1")
        if not enabled:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.root = ensure_dir(Path(config["paths"]["debug_root"]) / stamp)
        ensure_dir(self.root / "frames")
        ensure_dir(self.root / "crops")
        self.events_file = (self.root / "events.jsonl").open("a", encoding="utf-8")
        self._write_dashboard_html()
        if bool(config["debug"].get("dashboard", True)):
            self.start_dashboard(int(port or config["debug"]["port"]))

    def _write_dashboard_html(self):
        html = """<!doctype html>
<html><head><meta charset="utf-8"><title>TonyPi Debug</title>
<style>
body{font-family:Arial,sans-serif;margin:16px;background:#111;color:#eee}
img{max-width:49%;border:1px solid #555;margin:4px;vertical-align:top}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}
.card{background:#1c1c1c;border:1px solid #444;border-radius:6px;padding:10px}
.ok{color:#6ee27a}.bad{color:#ff8585}.warn{color:#ffd166}
.muted{color:#999}.pill{display:inline-block;border:1px solid #555;border-radius:999px;padding:1px 7px;margin:1px 4px 1px 0;background:#252525}
pre{background:#222;padding:10px;white-space:pre-wrap;max-height:360px;overflow:auto}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #444;padding:4px;text-align:left;vertical-align:top}
td pre{margin:0;max-height:120px;padding:6px;font-size:12px}
</style>
<script>
function esc(x){return String(x===null||x===undefined?'':x).replace(/[&<>]/g,s=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[s]));}
function fmt(x,d=1){if(x===null||x===undefined||x==='')return'-'; const n=Number(x); return Number.isFinite(n)?n.toFixed(d):'-';}
function clsForStatus(s){return s==='CHANGED'?'ok':(s==='FAILED'?'bad':(s==='ALREADY_TARGET'?'warn':''));}
function renderSummary(data){
  const pose=data.robot&&data.robot.pose?data.robot.pose:{};
  const plan=data.last_target_plan||{};
  const passby=data.active_passby_stop||data.last_passby_plan||{};
  const health=data.localization_health||{};
  const recovery=data.recovery||{};
  const lastRecovery=recovery.last||{};
  const targetScreen=data.target_screen||{};
  const interaction=data.interaction||{};
  const check=interaction.last_check||{};
  const routeScreens=(plan.route_screen_ids||[]).join(',');
  const passbyScreens=(passby.screen_ids||[]).join(',');
  const passbyPans=(passby.pan_angles||[]).join(',');
  document.getElementById('summary').innerHTML =
    '<b>mode</b>: '+esc(data.mode)+' &nbsp; <b>target</b>: '+esc(data.target_flower)+
    ' &nbsp; <b>physically changed</b>: '+esc(data.completed_count)+' &nbsp; <b>time</b>: '+esc(data.time_left_s)+'s<br>'+
    '<b>pose</b>: x='+esc(fmt(pose.x_cm))+
    ', y='+esc(fmt(pose.y_cm))+
    ', yaw='+esc(fmt(pose.yaw_deg))+
    ', conf='+esc(pose.confidence||'')+'<br>'+
    '<b>current target</b>: '+esc(targetScreen.screen_id||'-')+
    ' '+(targetScreen.status?`<span class="${clsForStatus(targetScreen.status)}">${esc(targetScreen.status)}</span>`:'')+'<br>'+
    '<b>plan</b>: score='+esc(plan.score||'-')+
    ', travel='+esc(plan.travel_cm||'-')+
    'cm, turn='+esc(plan.turn_deg||'-')+
    'deg, turnCost='+esc(plan.turn_cost||0)+
    ', routeBonus='+esc(plan.route_bonus||0)+
    ', route screens='+esc(routeScreens||'-')+'<br>'+
    '<b>passby stop</b>: screens='+esc(passbyScreens||'-')+
    ', sides='+esc((passby.sides||[]).join(',')||'-')+
    ', pans='+esc(passbyPans||'-')+
    ', xy='+esc(passby.xy?passby.xy.join(','):'-')+'<br>'+
    '<b>tag health</b>: noTagScans='+esc(health.consecutive_no_tag_scans||0)+
    ', locFailures='+esc(health.consecutive_localize_failures||0)+
    ', sinceTag='+esc(fmt(health.seconds_since_any_tag,1))+'s'+
    ', sinceLoc='+esc(fmt(health.seconds_since_localize,1))+'s<br>'+
    '<b>recovery</b>: count='+esc(recovery.count||0)+
    ', last='+esc(lastRecovery.reason||'-')+
    ', outward='+esc(recovery.facing_outside)+
    ', exitAhead='+esc(fmt(recovery.field_exit_ahead_cm,1))+'cm'+
    ', noProgress='+esc(recovery.no_progress_count||0)+'<br>'+
    '<b>interaction</b>: phase='+esc(interaction.phase||'idle')+
    ', ready='+esc(interaction.ready)+
    ', leftHand='+esc(interaction.left_hand_lifted)+
    ', distance='+esc(fmt(check.distance_cm,1))+'cm'+
    ', distanceErr='+esc(fmt(check.distance_error_cm,1))+'cm'+
    ', yawErr='+esc(fmt(check.yaw_error_deg,1))+'deg'+
    ', lateralErr='+esc(fmt(check.lateral_error_cm,1))+'cm'+
    ', blockers='+esc((check.reasons||[]).join(',')||'-');
}
function renderVotes(summary){
  if(!summary||!summary.screens){document.getElementById('votes').innerHTML='No vote data yet.';return;}
  let rows='';
  Object.values(summary.screens).sort((a,b)=>a.screen_id-b.screen_id).forEach(s=>{
    const best=s.best?`${esc(s.best.flower)} (${s.best.count}x, conf ${s.best.avg_confidence})`:'-';
    const votes=Object.entries(s.votes||{}).map(([k,v])=>`${esc(k)}:${v.count}`).join(', ');
    const obs=(s.observations||[]).slice(-5).map(o=>{
      const why=o.reject_reason||o.error||'';
      return `${esc(o.pan)} ${esc(o.flower||'-')} ${esc(o.confidence||'')} ${esc(why)}`;
    }).join('<br>');
    rows+=`<tr><td>${s.screen_id}</td><td>${best}</td><td>${esc(s.decision)}</td><td>${esc(votes)}</td><td>${(s.observations||[]).length}</td><td>${obs}</td></tr>`;
  });
  document.getElementById('votes').innerHTML =
    `<div><span class="pill">reason=${esc(summary.reason||'-')}</span><span class="pill">frames=${esc(summary.vote_frames)}</span><span class="pill">pans=${esc((summary.pan_angles||[]).join(','))}</span><span class="pill">min_votes=${esc(summary.min_votes)}</span><span class="pill">min_conf=${esc(summary.min_confidence)}</span></div>`+
    '<table><tr><th>screen</th><th>best</th><th>decision</th><th>votes</th><th>obs</th><th>last observations</th></tr>'+rows+'</table>';
}
function renderInteraction(result,data){
  const logPath=data&&data.interaction_log_path?data.interaction_log_path:'';
  let html=`<div><b>interaction log</b>: ${esc(logPath||'not created yet')}</div>`;
  if(!result){document.getElementById('interaction').innerHTML=html+'<div>No Worker request yet.</div>';return;}
  const cls=result.success?'ok':'bad';
  html +=
    `<div class="${cls}"><b>success</b>: ${esc(result.success)} &nbsp; <b>simulated</b>: ${esc(result.simulated)}</div>`+
    `<div><b>screen</b>: ${esc(result.screen_id)} &nbsp; <b>worker</b>: ${esc(result.worker_id)} &nbsp; <b>from</b>: ${esc(result.from_flower)} &nbsp; <b>to</b>: ${esc(result.to_flower)}</div>`+
    `<div><b>pose check</b>: ${esc(JSON.stringify(result.interaction_check||{}))}</div>`+
    `<div><b>response</b>: ${esc(JSON.stringify(result.response||{}))}</div>`+
    `<div><b>error</b>: ${esc(result.error)}</div>`;
  const recent=(data&&data.recent_interactions)||[];
  if(recent.length){
    let rows='';
    recent.slice().reverse().forEach(r=>{
      rows+=`<tr><td>${esc(r.screen_id)}</td><td>${esc(r.worker_id)}</td><td>${esc(r.from_flower)}</td><td>${esc(r.to_flower)}</td><td>${esc(JSON.stringify(r.response||{})||r.error)}</td></tr>`;
    });
    html += '<table><tr><th>screen</th><th>worker</th><th>from</th><th>to</th><th>result</th></tr>'+rows+'</table>';
  }
  document.getElementById('interaction').innerHTML = html;
}
function renderScreens(data){
  const screens=data.screens||{};
  let rows='';
  Object.values(screens).sort((a,b)=>a.screen_id-b.screen_id).forEach(s=>{
    const status=s.status||'';
    rows+=`<tr><td>${esc(s.screen_id)}</td><td>${esc(s.worker_id||'-')}</td><td class="${clsForStatus(status)}">${esc(status)}</td><td>${esc(s.attempts)}</td><td>${esc(s.last_classification||'-')}</td><td>${esc(fmt(s.last_confidence,3))}</td><td>${esc((s.observation_xy||[]).join(','))}</td><td>${esc((s.interaction_xy||[]).join(','))} @ ${esc(fmt(s.interaction_yaw_deg,1))}deg</td><td>${esc((s.notes||[]).join('; '))}</td></tr>`;
  });
  document.getElementById('screens').innerHTML =
    '<table><tr><th>screen</th><th>worker</th><th>status</th><th>attempts</th><th>flower</th><th>conf</th><th>observation XY</th><th>interaction XY/yaw</th><th>notes</th></tr>'+rows+'</table>';
}
function renderEvents(events){
  events=events||[];
  if(!events.length){document.getElementById('events').innerHTML='No events yet.';return;}
    const important=new Set(['flower_observed','interaction_alignment_check','interaction_safety_gate_blocked','interaction_changed','interaction_not_changed','interaction_exception','left_hand_lifted','worker_request_sent','worker_response','already_target','classification_failed','classification_low_confidence','target_selected','target_body_reaim','target_body_reaim_skipped','target_not_completed_after_arrival','navigate_failed','near_wall_recover','front_obstacle_recover','forward_blocked_by_map','forward_no_progress','visual_forward_no_progress','visual_progress_check_inconclusive','visual_forward_progress_restored','no_tag_recovery_triggered','recovery_start','recovery_backoff_localize_attempt','recovery_done','translation_step','turn_last_resort','turn_last_resort_noop','scan_after_turn_done','scan_after_turn_failed','boundary_pan_filtered','boundary_safe_turn','boundary_recovery_target_selected','boundary_blind_nav_start','boundary_blind_nav_step','boundary_blind_nav_arrived','boundary_blind_nav_failed','harvest_skipped_boundary_outward','localize_skipped_boundary_outward','localize_harvest_done','localize_harvest_failed','head_recenter_after_scan','head_recenter_failed','initial_discovery_scan_start','initial_discovery_scan_turn','initial_discovery_scan_relocalize_start','initial_discovery_scan_relocalize_done','initial_discovery_scan_done','opportunistic_harvest','route_passby_stop_selected','route_passby_scan_start','route_passby_scan_done','action','pose_update']);
  let rows='';
  events.slice().reverse().forEach(e=>{
    const detail=Object.assign({}, e); delete detail.t; delete detail.event;
    const cls=important.has(e.event)?'warn':'';
    rows+=`<tr class="${cls}"><td>${esc(new Date((e.t||0)*1000).toLocaleTimeString())}</td><td>${esc(e.event)}</td><td><pre>${esc(JSON.stringify(detail,null,2))}</pre></td></tr>`;
  });
  document.getElementById('events').innerHTML =
    '<table><tr><th>time</th><th>event</th><th>detail</th></tr>'+rows+'</table>';
}
function renderEventsFromStateOrLog(data){
  const inline=data&&data.recent_events?data.recent_events:[];
  if(inline.length){renderEvents(inline);return;}
  fetch('events.jsonl?t='+Date.now()).then(r=>r.text()).then(text=>{
    const events=text.trim().split('\\n').slice(-80).map(line=>{
      try{return JSON.parse(line);}catch(e){return null;}
    }).filter(Boolean);
    renderEvents(events);
  }).catch(()=>renderEvents([]));
}
function refresh(){
  document.getElementById('ann').src='latest_annotated.jpg?t='+Date.now();
  document.getElementById('map').src='latest_map.jpg?t='+Date.now();
  fetch('latest_state.json?t='+Date.now()).then(r=>r.json()).then(data=>{
    renderSummary(data); renderVotes(data.last_vote_summary); renderInteraction(data.latest_interaction,data);
    renderScreens(data); renderEventsFromStateOrLog(data);
    document.getElementById('state').textContent=JSON.stringify(data,null,2);
  }).catch(()=>{});
}
setInterval(refresh,800);
</script></head>
<body onload="refresh()">
<h2>TonyPi Debug</h2>
<div><img id="ann" src="latest_annotated.jpg"><img id="map" src="latest_map.jpg"></div>
<div class="grid">
  <div class="card"><h3>State</h3><div id="summary"></div></div>
  <div class="card"><h3>Physical Interaction / Worker</h3><div id="interaction"></div></div>
  <div class="card" style="grid-column:1/3"><h3>Last Vote Summary</h3><div id="votes"></div></div>
  <div class="card" style="grid-column:1/3"><h3>Screen Status</h3><div id="screens"></div></div>
  <div class="card" style="grid-column:1/3"><h3>Recent Events</h3><div id="events"></div></div>
  <div class="card" style="grid-column:1/3"><h3>Raw State JSON</h3><pre id="state"></pre></div>
</div>
</body></html>
"""
        (self.root / "index.html").write_text(html, encoding="utf-8")

    def start_dashboard(self, port: int):
        handler = functools.partial(SimpleHTTPRequestHandler, directory=str(self.root))
        self.httpd = HTTPServer((self.host, port), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        display_host = "127.0.0.1" if self.host in ("", "0.0.0.0") else self.host
        print("[debug] dashboard: http://{}:{}".format(display_host, port))

    def event(self, name: str, **data):
        payload = {"t": time.time(), "event": name}
        payload.update(data)
        print("[{}] {}".format(name, data))
        self.recent_events.append(payload)
        self.recent_events = self.recent_events[-80:]
        if not self.enabled or self.events_file is None:
            return
        self.events_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.events_file.flush()

    def save_image(self, name: str, image, force: bool = False):
        if not self.enabled or image is None or not bool(self.config["debug"].get("save_images", True)):
            return
        import cv2

        now = time.monotonic()
        if not force and now - self.last_image_s < float(self.config["debug"]["image_min_interval_s"]):
            return
        self.last_image_s = now
        cv2.imwrite(str(self.root / name), image)

    def save_crop(self, screen_id: int, crop, suffix: str):
        if not self.enabled or crop is None:
            return
        import cv2

        cv2.imwrite(str(self.root / "crops" / "screen_{}_{}.jpg".format(screen_id, suffix)), crop)

    def state(self, data: Dict[str, Any]):
        if not self.enabled:
            return
        data = dict(data)
        data["recent_events"] = self.recent_events[-80:]

        with (self.root / "latest_state.json").open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def render_map(self, map_model, pose=None, path=None, target_screen=None, scan_stops=None):
        if not self.enabled:
            return
        import cv2

        scale = 2.0
        img = np.full((int(map_model.height_cm * scale), int(map_model.width_cm * scale), 3), 245, dtype=np.uint8)
        for screen in map_model.screens.values():
            color = (180, 180, 180)
            status = str(getattr(screen.status, "value", screen.status))
            if status == "CHANGED":
                color = (80, 190, 80)
            elif status == "ALREADY_TARGET":
                color = (80, 190, 230)
            elif status == "FAILED":
                color = (80, 80, 220)
            if target_screen is not None and screen.screen_id == target_screen.screen_id:
                color = (40, 80, 240)
            c = self._map_pt(screen.center_xy, scale, img.shape[0])
            observation = self._map_pt(screen.observation_xy, scale, img.shape[0])
            interaction = self._map_pt(screen.interaction_xy, scale, img.shape[0])
            reader = self._map_pt(screen.reader_xy, scale, img.shape[0])
            cv2.circle(img, c, 4, color, -1)
            cv2.circle(img, observation, 3, (0, 160, 220), -1)
            cv2.circle(img, interaction, 3, (220, 80, 40), -1)
            cv2.circle(img, reader, 2, (180, 0, 180), -1)
            cv2.putText(img, str(screen.screen_id), (c[0] + 4, c[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        if path:
            pts = [self._map_pt(pt, scale, img.shape[0]) for pt in path]
            for p0, p1 in zip(pts, pts[1:]):
                cv2.line(img, p0, p1, (255, 120, 0), 2)
        if scan_stops:
            for stop in scan_stops:
                xy = stop.get("xy")
                if not xy:
                    continue
                p = self._map_pt(xy, scale, img.shape[0])
                cv2.circle(img, p, 5, (220, 40, 220), -1)
                label = "S{}".format(",".join(str(s) for s in stop.get("screen_ids", [])[:3]))
                cv2.putText(img, label, (p[0] + 5, p[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 0, 160), 1)
        if pose is not None:
            p = self._map_pt((pose.x_cm, pose.y_cm), scale, img.shape[0])
            cv2.circle(img, p, 6, (0, 0, 0), -1)
            yaw = np.radians(pose.yaw_deg)
            q = (int(p[0] + 18 * np.cos(yaw)), int(p[1] - 18 * np.sin(yaw)))
            cv2.line(img, p, q, (0, 0, 0), 2)
        cv2.imwrite(str(self.root / "latest_map.jpg"), img)

    def _map_pt(self, xy, scale, height_px):
        return int(xy[0] * scale), int(height_px - xy[1] * scale)

    def close(self):
        if self.httpd is not None:
            self.httpd.shutdown()
        if self.events_file is not None:
            self.events_file.close()
