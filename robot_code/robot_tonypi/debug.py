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
<html lang="zh-CN"><head><meta charset="utf-8"><title>TonyPi 调试面板 / Debug Dashboard</title>
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
const statusZh={UNKNOWN:'未知',NEEDS_CHANGE:'需要换花',INTERACTING:'交互中',CHANGED:'已换花',ALREADY_TARGET:'已是目标花',FAILED:'失败'};
const modeZh={mission:'完整任务',localize:'仅定位',harvest:'仅扫描识别'};
const phaseZh={idle:'空闲',transaction_start:'交互开始',stand:'站立',left_hand_lifted:'左手已举起',worker_request_sent:'已发送 Worker 请求',worker_response:'收到 Worker 响应',transaction_end:'交互结束'};
const reasonZh={target_confirmation_missing:'缺少25cm目标确认',confirmation_screen_mismatch:'确认屏幕不匹配',confirmation_tag_mismatch:'确认Tag不匹配',confirmation_binding_missing:'确认时Tag与屏幕未绑定',visual_authorization_missing:'缺少目标视觉授权',authorization_screen_mismatch:'授权屏幕不匹配',authorization_tag_mismatch:'授权 Tag 不匹配',authorization_binding_missing:'Tag 与屏幕未绑定',authorization_flower_mismatch:'授权花朵结果不匹配',flower_unknown:'花朵未知',already_target:'已是目标花',flower_changed_since_capture:'拍摄后花朵状态改变',target_lock_mismatch:'当前目标锁不匹配',target_not_arrived:'当前目标未到达'};
const eventZh={mission_state:'任务状态',navigation_mode:'导航模式',target_direct_approach_action:'近目标直达动作',target_direct_recovery_suppressed:'末段抑制近墙恢复',transit_bindings_updated:'途中几何绑定更新',arrived_at_target:'已到达当前目标',target_tag_and_screen_confirmed:'目标Tag与屏幕确认',target_visual_authorized:'FPGA视觉授权成功',target_visual_confirmation_failed:'目标视觉确认失败',target_classification_failed:'当前目标分类失败',classifier_gate_blocked:'分类入口阻止',target_final_forward_started:'开始前进13cm',target_final_forward_done:'已前进13cm',target_final_forward_failed:'13cm动作失败',screen_needs_change:'屏幕需要换花',already_target:'已是目标花',target_selected:'已选择最近目标',interaction_safety_gate_blocked:'交互授权阻止',interaction_changed:'换花成功',interaction_not_changed:'换花未成功',interaction_exception:'交互异常',left_hand_lifted:'左手已举起',worker_request_sent:'已发送 Worker 请求',worker_response:'收到 Worker 响应',navigate_failed:'导航失败',localize_failed:'定位失败',pose_update:'定位已更新',action:'执行动作',recovery_start:'开始恢复',recovery_done:'恢复完成',turn_no_progress:'转向无进展',suspect_stale_pose_after_turn:'转向后疑似旧位姿',turn_direction_conflict:'转向方向冲突',scan_after_turn_pose_rejected:'拒绝转向后视觉位姿',turn_progress_relocalize:'转向进展强制重定位',turn_progress_failed:'转向进展失败',turn_progress_restored:'转向进展恢复'};
function bi(value,dict){const raw=String(value===null||value===undefined?'':value); return dict[raw]?`${dict[raw]} / ${raw}`:raw;}
function reasonsBi(values){return (values||[]).map(x=>bi(x,reasonZh)).join(',')||'-';}
function renderSummary(data){
  const pose=data.robot&&data.robot.pose?data.robot.pose:{};
  const plan=data.last_target_plan||{};
  const health=data.localization_health||{};
  const recovery=data.recovery||{};
  const lastRecovery=recovery.last||{};
  const targetScreen=data.target_screen||{};
  const interaction=data.interaction||{};
  const check=interaction.last_check||{};
  const authorization=data.visual_authorization||{};
  const confirmation=data.target_visual_confirmation||{};
  const bindings=data.transit_bindings||{};
  document.getElementById('summary').innerHTML =
    '<b>运行模式 / Mode</b>: '+esc(bi(data.mode,modeZh))+' &nbsp; <b>目标花 / Target</b>: '+esc(data.target_flower)+
    ' &nbsp; <b>实际换花数 / Changed</b>: '+esc(data.completed_count)+' &nbsp; <b>已处理 / Processed</b>: '+esc(data.processed_count)+' &nbsp; <b>剩余时间 / Time left</b>: '+esc(data.time_left_s)+'s<br>'+
    '<b>机器人位姿 / Pose</b>: x='+esc(fmt(pose.x_cm))+
    ', y='+esc(fmt(pose.y_cm))+
    ', yaw='+esc(fmt(pose.yaw_deg))+
    ', 置信度/conf='+esc(pose.confidence||'')+'<br>'+
    '<b>任务状态 / Mission state</b>: '+esc(data.mission_state||'-')+'<br>'+
    '<b>当前目标 / Current target</b>: Tag '+esc(data.current_target_tag_id||'-')+', Screen '+esc(targetScreen.screen_id||'-')+
    ' '+(targetScreen.status?`<span class="${clsForStatus(targetScreen.status)}">${esc(bi(targetScreen.status,statusZh))}</span>`:'')+'<br>'+
    '<b>最近目标选择 / Nearest selection</b>: 距离/distance='+esc(fmt(data.current_target_distance_cm,1))+'cm, 25cm target='+esc((plan.task_target_xy||[]).join(','))+', yaw='+esc(fmt(plan.task_target_yaw_deg,1))+'°, face='+esc(plan.surface_face||'-')+', normal='+esc((plan.cardinal_normal_xy||[]).join(','))+', 剩余/remaining='+esc((data.remaining_target_ids||[]).join(','))+'<br>'+
    '<b>25cm确认与分类授权 / Confirmation & authorization</b>: arrived='+esc(data.arrived_at_target)+', confirmedTag='+esc(confirmation.tag_id||'-')+', confirmedScreen='+esc(confirmation.screen_id||'-')+', bound='+esc(confirmation.binding_ok)+', forward13cm='+esc(data.final_forward_executed)+', classifierAllowed='+esc(data.classifier_allowed)+', flower='+esc(authorization.flower||'-')+', confidence='+esc(fmt(authorization.confidence,3))+'<br>'+
    '<b>途中几何绑定 / Transit bindings</b>: '+esc(Object.keys(bindings).join(',')||'-')+'（只框屏幕并绑定左上 Tag，不分类 / geometry only）<br>'+
    '<b>Tag/定位健康 / Localization health</b>: 无Tag扫描/noTagScans='+esc(health.consecutive_no_tag_scans||0)+
    ', 定位失败/locFailures='+esc(health.consecutive_localize_failures||0)+
    ', 距上次Tag/sinceTag='+esc(fmt(health.seconds_since_any_tag,1))+'s'+
    ', 距上次定位/sinceLoc='+esc(fmt(health.seconds_since_localize,1))+'s<br>'+
    '<b>恢复 / Recovery</b>: 次数/count='+esc(recovery.count||0)+
    ', 最近原因/last='+esc(lastRecovery.reason||'-')+
    ', 朝向场外/outward='+esc(recovery.facing_outside)+
    ', 前方出界距离/exitAhead='+esc(fmt(recovery.field_exit_ahead_cm,1))+'cm'+
    ', 无进展/noProgress='+esc(recovery.no_progress_count||0)+'<br>'+
    '<b>实体交互 / Interaction</b>: 阶段/phase='+esc(bi(interaction.phase||'idle',phaseZh))+
    ', 视觉授权/authorized='+esc(interaction.ready)+
    ', 左手举起/leftHand='+esc(interaction.left_hand_lifted)+
    ', 阻止原因/blockers='+esc(reasonsBi(check.reasons));
}
function renderVotes(summary){
  if(!summary||!summary.screens){document.getElementById('votes').innerHTML='暂无投票数据 / No vote data yet.';return;}
  let rows='';
  Object.values(summary.screens).sort((a,b)=>a.screen_id-b.screen_id).forEach(s=>{
    const best=s.best?`${esc(s.best.flower)} (${s.best.count}次/x, 置信度/conf ${s.best.avg_confidence})`:'-';
    const votes=Object.entries(s.votes||{}).map(([k,v])=>`${esc(k)}:${v.count}`).join(', ');
    const obs=(s.observations||[]).slice(-5).map(o=>{
      const why=o.reject_reason||o.error||'';
      return `${esc(o.pan)} ${esc(o.flower||'-')} ${esc(o.confidence||'')} ${esc(why)}`;
    }).join('<br>');
    rows+=`<tr><td>${s.screen_id}</td><td>${best}</td><td>${esc(s.decision)}</td><td>${esc(votes)}</td><td>${(s.observations||[]).length}</td><td>${obs}</td></tr>`;
  });
  document.getElementById('votes').innerHTML =
    `<div><span class="pill">原因/reason=${esc(summary.reason||'-')}</span><span class="pill">帧数/frames=${esc(summary.vote_frames)}</span><span class="pill">头部角度/pans=${esc((summary.pan_angles||[]).join(','))}</span><span class="pill">最低票数/min_votes=${esc(summary.min_votes)}</span><span class="pill">最低置信度/min_conf=${esc(summary.min_confidence)}</span></div>`+
    '<table><tr><th>屏幕 / Screen</th><th>最佳结果 / Best</th><th>决策 / Decision</th><th>票数 / Votes</th><th>观察数 / Obs</th><th>最近观察 / Last observations</th></tr>'+rows+'</table>';
}
function renderInteraction(result,data){
  const logPath=data&&data.interaction_log_path?data.interaction_log_path:'';
  let html=`<div><b>交互日志 / Interaction log</b>: ${esc(logPath||'尚未创建 / not created yet')}</div>`;
  if(!result){document.getElementById('interaction').innerHTML=html+'<div>尚无 Worker 请求 / No Worker request yet.</div>';return;}
  const cls=result.success?'ok':'bad';
  html +=
    `<div class="${cls}"><b>成功 / Success</b>: ${esc(result.success)} &nbsp; <b>模拟 / Simulated</b>: ${esc(result.simulated)}</div>`+
    `<div><b>屏幕 / Screen</b>: ${esc(result.screen_id)} &nbsp; <b>Worker</b>: ${esc(result.worker_id)} &nbsp; <b>原花 / From</b>: ${esc(result.from_flower)} &nbsp; <b>目标花 / To</b>: ${esc(result.to_flower)}</div>`+
    `<div><b>视觉授权检查 / Visual authorization</b>: ${esc(JSON.stringify(result.interaction_check||{}))}</div>`+
    `<div><b>响应 / Response</b>: ${esc(JSON.stringify(result.response||{}))}</div>`+
    `<div><b>错误 / Error</b>: ${esc(result.error)}</div>`;
  const recent=(data&&data.recent_interactions)||[];
  if(recent.length){
    let rows='';
    recent.slice().reverse().forEach(r=>{
      rows+=`<tr><td>${esc(r.screen_id)}</td><td>${esc(r.worker_id)}</td><td>${esc(r.from_flower)}</td><td>${esc(r.to_flower)}</td><td>${esc(JSON.stringify(r.response||{})||r.error)}</td></tr>`;
    });
    html += '<table><tr><th>屏幕 / Screen</th><th>Worker</th><th>原花 / From</th><th>目标花 / To</th><th>结果 / Result</th></tr>'+rows+'</table>';
  }
  document.getElementById('interaction').innerHTML = html;
}
function renderScreens(data){
  const screens=data.screens||{};
  let rows='';
  Object.values(screens).sort((a,b)=>a.screen_id-b.screen_id).forEach(s=>{
    const status=s.status||'';
    rows+=`<tr><td>${esc(s.screen_id)}</td><td>${esc(s.worker_id||'-')}</td><td class="${clsForStatus(status)}">${esc(bi(status,statusZh))}</td><td>${esc(s.attempts)}</td><td>${esc(s.last_classification||'-')}</td><td>${esc(fmt(s.last_confidence,3))}</td><td>${esc((s.task_target_xy||s.target_xy||[]).join(','))} @ ${esc(fmt(s.task_target_yaw_deg===null?s.interaction_yaw_deg:s.task_target_yaw_deg,1))}deg</td><td>${esc(s.surface_face||'-')} / ${esc((s.cardinal_normal_xy||[]).join(','))}</td><td>${esc((s.notes||[]).join('; '))}</td></tr>`;
  });
  document.getElementById('screens').innerHTML =
    '<table><tr><th>屏幕 / Screen</th><th>Worker</th><th>状态 / Status</th><th>尝试 / Attempts</th><th>花朵 / Flower</th><th>置信度 / Conf</th><th>25cm唯一目标 / Target pose</th><th>目标面 / Face-normal</th><th>备注 / Notes</th></tr>'+rows+'</table>';
}
function renderEvents(events){
  events=events||[];
  if(!events.length){document.getElementById('events').innerHTML='暂无事件 / No events yet.';return;}
    const important=new Set(['mission_state','navigation_mode','target_direct_approach_action','target_direct_recovery_suppressed','transit_bindings_updated','arrived_at_target','target_tag_and_screen_confirmed','target_visual_authorized','target_visual_confirmation_failed','target_classification_failed','classifier_gate_blocked','interaction_safety_gate_blocked','target_final_forward_started','target_final_forward_done','target_final_forward_failed','interaction_changed','interaction_not_changed','interaction_exception','left_hand_lifted','worker_request_sent','worker_response','already_target','classification_failed','classification_low_confidence','target_selected','target_not_completed_after_arrival','navigate_failed','target_failed','mission_complete','near_wall_recover','front_obstacle_recover','forward_blocked_by_map','forward_no_progress','visual_forward_no_progress','visual_progress_check_inconclusive','visual_forward_progress_restored','no_tag_recovery_triggered','recovery_start','recovery_backoff_localize_attempt','recovery_done','translation_step','turn_last_resort','turn_last_resort_noop','scan_after_turn_done','scan_after_turn_failed','turn_no_progress','suspect_stale_pose_after_turn','turn_direction_conflict','scan_after_turn_pose_rejected','turn_progress_relocalize','turn_progress_failed','turn_progress_restored','boundary_pan_filtered','boundary_safe_turn','boundary_recovery_target_selected','boundary_blind_nav_start','boundary_blind_nav_step','boundary_blind_nav_arrived','boundary_blind_nav_failed','localize_skipped_boundary_outward','head_recenter_after_scan','head_recenter_failed','action','pose_update']);
  let rows='';
  events.slice().reverse().forEach(e=>{
    const detail=Object.assign({}, e); delete detail.t; delete detail.event;
    const cls=important.has(e.event)?'warn':'';
    rows+=`<tr class="${cls}"><td>${esc(new Date((e.t||0)*1000).toLocaleTimeString())}</td><td>${esc(bi(e.event,eventZh))}</td><td><pre>${esc(JSON.stringify(detail,null,2))}</pre></td></tr>`;
  });
  document.getElementById('events').innerHTML =
    '<table><tr><th>时间 / Time</th><th>事件 / Event</th><th>详情 / Detail</th></tr>'+rows+'</table>';
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
<h2>TonyPi 调试面板 / Debug Dashboard</h2>
<div class="muted">中文用于现场阅读，英文字段保留用于日志定位和代码检索。/ Chinese is provided for field use; English identifiers remain for log and code lookup.</div>
<div><img id="ann" src="latest_annotated.jpg"><img id="map" src="latest_map.jpg"></div>
<div class="grid">
  <div class="card"><h3>运行状态 / State</h3><div id="summary"></div></div>
  <div class="card"><h3>实体交互与 Worker / Physical Interaction</h3><div id="interaction"></div></div>
  <div class="card" style="grid-column:1/3"><h3>最近投票汇总 / Last Vote Summary</h3><div id="votes"></div></div>
  <div class="card" style="grid-column:1/3"><h3>屏幕状态 / Screen Status</h3><div id="screens"></div></div>
  <div class="card" style="grid-column:1/3"><h3>最近事件 / Recent Events</h3><div id="events"></div></div>
  <div class="card" style="grid-column:1/3"><h3>原始状态 JSON / Raw State JSON</h3><pre id="state"></pre></div>
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
            target = self._map_pt(screen.target_xy, scale, img.shape[0])
            reader = self._map_pt(screen.reader_xy, scale, img.shape[0])
            cv2.circle(img, c, 4, color, -1)
            cv2.circle(img, target, 4, (220, 80, 40), -1)
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
