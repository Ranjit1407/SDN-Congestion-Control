#!/usr/bin/env python3
# Controller_UnifiedController.py
#
# Ryu OF1.3 controller that:
# 1) Acts as a learning switch (L2).
# 2) Polls port stats and appends to logs/stats_log.csv.
# 3) Calls an AgentManager to pick an action per (dpid,port).
# 4) ENFORCES actions via OpenFlow 1.3:
#       0 = noop
#       1 = rate-limit ingress port with a meter (drop band)
#       2 = drop all ingress traffic on that port (temporary)
#       3 = (stub) "reroute": log + FLOOD (placeholder)
#
# This version prepends the project root to sys.path so imports like
# "from agents.multi.agent_manager import AgentManager" always work.

import os
import sys
import csv
import time

# ----- Ensure project root is on sys.path -----
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # ~/Documents/project1
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ----------------------------------------------

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import (
    MAIN_DISPATCHER, DEAD_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
)
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types
from ryu.lib import hub

from agents.multi.agent_manager import AgentManager  # now resolvable

# --- paths and polling ---
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
LOG_PATH = os.path.join(LOG_DIR, 'stats_log.csv')
POLL_INTERVAL = 1.0  # seconds

# --- action ids ---
ACT_NOOP    = 0
ACT_METER   = 1
ACT_DROP    = 2
ACT_REROUTE = 3   # stub

# --- meter params (adjust) ---
DEFAULT_LIMIT_KBPS = 5000       # 5 Mbps cap per ingress port when action=1
DEFAULT_IDLE_TO    = 20         # seconds for temporary rules


class UnifiedController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(UnifiedController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}          # dpid -> {mac: port}
        self.datapaths = {}            # dpid -> datapath
        self.last_stats = {}           # (dpid,port) -> last sample
        self.meter_installed = set()   # {(dpid, meter_id)}
        self.monitor_thread = hub.spawn(self._monitor)

        # agent
        self.agent_manager = AgentManager()
        self.agent_manager.load_models()

        # ensure logs csv exists
        os.makedirs(LOG_DIR, exist_ok=True)
        if not os.path.exists(LOG_PATH):
            with open(LOG_PATH, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['timestamp', 'dpid', 'port_no',
                            'rx_bytes', 'tx_bytes', 'rx_packets', 'tx_packets',
                            'duration_sec'])

    # ------------------------------
    # Switch connect / table-miss
    # ------------------------------
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        dpid = dp.id
        self.logger.info("Switch connected: dpid=%s", dpid)
        self.datapaths[dpid] = dp
        self._install_table_miss(dp)

    def _install_table_miss(self, dp):
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=dp, priority=0, match=match, instructions=inst)
        dp.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
        elif ev.state == DEAD_DISPATCHER:
            self.datapaths.pop(dp.id, None)

    # ------------------------------
    # Learning switch
    # ------------------------------
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in(self, ev):
        msg = ev.msg
        dp = msg.datapath
        dpid = dp.id
        ofp = dp.ofproto
        parser = dp.ofproto_parser

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        in_port = msg.match['in_port']
        dst = eth.dst
        src = eth.src

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofp.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofp.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
            mod = parser.OFPFlowMod(datapath=dp, priority=1,
                                    match=match, instructions=inst,
                                    idle_timeout=60, hard_timeout=0)
            dp.send_msg(mod)

        out = parser.OFPPacketOut(datapath=dp, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions,
                                  data=None if msg.buffer_id != ofp.OFP_NO_BUFFER else msg.data)
        dp.send_msg(out)

    # ------------------------------
    # Stats polling + agent + enforcement
    # ------------------------------
    def _monitor(self):
        while True:
            for dp in list(self.datapaths.values()):
                self._request_port_stats(dp)
            hub.sleep(POLL_INTERVAL)

    def _request_port_stats(self, dp):
        parser = dp.ofproto_parser
        req = parser.OFPPortStatsRequest(dp, 0, dp.ofproto.OFPP_ANY)
        dp.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        now = time.time()
        dp = ev.msg.datapath
        dpid = dp.id
        parser = dp.ofproto_parser

        rows = []
        for stat in ev.msg.body:
            port_no = stat.port_no
            key = (dpid, port_no)
            cur = {
                'timestamp': now,
                'dpid': dpid,
                'port_no': port_no,
                'rx_bytes': stat.rx_bytes,
                'tx_bytes': stat.tx_bytes,
                'rx_packets': getattr(stat, 'rx_packets', 0),
                'tx_packets': getattr(stat, 'tx_packets', 0),
                'duration_sec': getattr(stat, 'duration_sec', 0)
            }
            rows.append([
                f"{now:.6f}", dpid, port_no,
                cur['rx_bytes'], cur['tx_bytes'], cur['rx_packets'], cur['tx_packets'],
                cur['duration_sec']
            ])

            prev = self.last_stats.get(key)
            if prev:
                dt = max(1e-9, now - prev['timestamp'])
                d_tx = cur['tx_bytes'] - prev['tx_bytes']
                d_rx = cur['rx_bytes'] - prev['rx_bytes']
                if d_tx >= 0 and d_rx >= 0:
                    tx_bps = (d_tx * 8.0) / dt
                    rx_bps = (d_rx * 8.0) / dt

                    action = self.agent_manager.act(dpid, port_no, tx_bps, rx_bps, explore=False)
                    self.logger.info("Agent action dpid=%s port=%s tx=%.2fbps rx=%.2fbps -> %s",
                                     dpid, port_no, tx_bps, rx_bps, action)
                    try:
                        self._enforce_action(dp, port_no, action)
                    except Exception as e:
                        self.logger.error("Enforce error dpid=%s port=%s: %s", dpid, port_no, e)

            self.last_stats[key] = cur

        # append raw stats
        try:
            with open(LOG_PATH, 'a', newline='') as f:
                w = csv.writer(f)
                w.writerows(rows)
        except Exception as e:
            self.logger.error("Failed to write stats: %s", e)

    # ------------------------------
    # Enforcement helpers
    # ------------------------------
    def _enforce_action(self, dp, in_port, action_id):
        ofp = dp.ofproto
        parser = dp.ofproto_parser

        if action_id == 0:  # NOOP
            return

        elif action_id == 1:  # METER
            meter_id = 1000 + int(in_port)
            self._ensure_meter(dp, meter_id, rate_kbps=DEFAULT_LIMIT_KBPS)
            match = parser.OFPMatch(in_port=in_port)
            inst = [
                parser.OFPInstructionMeter(meter_id),
                parser.OFPInstructionActions(
                    ofp.OFPIT_APPLY_ACTIONS,
                    [parser.OFPActionOutput(ofp.OFPP_NORMAL)]
                )
            ]
            mod = parser.OFPFlowMod(datapath=dp, priority=300,
                                    match=match, instructions=inst,
                                    idle_timeout=DEFAULT_IDLE_TO, hard_timeout=0)
            dp.send_msg(mod)
            self.logger.info("Installed meter flow dpid=%s in_port=%s meter_id=%s limit=%skbps",
                             dp.id, in_port, meter_id, DEFAULT_LIMIT_KBPS)

        elif action_id == 2:  # DROP
            match = parser.OFPMatch(in_port=in_port)
            inst = []  # drop
            mod = parser.OFPFlowMod(datapath=dp, priority=400,
                                    match=match, instructions=inst,
                                    idle_timeout=DEFAULT_IDLE_TO, hard_timeout=0)
            dp.send_msg(mod)
            self.logger.warning("Installed DROP on dpid=%s in_port=%s (idle_timeout=%s)",
                                dp.id, in_port, DEFAULT_IDLE_TO)

        elif action_id == 3:  # REROUTE (stub)
            match = parser.OFPMatch(in_port=in_port)
            inst = [
                parser.OFPInstructionActions(
                    ofp.OFPIT_APPLY_ACTIONS,
                    [parser.OFPActionOutput(ofp.OFPP_FLOOD)]
                )
            ]
            mod = parser.OFPFlowMod(datapath=dp, priority=200,
                                    match=match, instructions=inst,
                                    idle_timeout=DEFAULT_IDLE_TO, hard_timeout=0)
            dp.send_msg(mod)
            self.logger.info("REROUTE stub on dpid=%s in_port=%s (FLOOD)", dp.id, in_port)

    def _ensure_meter(self, dp, meter_id, rate_kbps):
        """Create a drop-band meter if not already installed."""
        key = (dp.id, meter_id)
        if key in getattr(self, 'meter_installed', set()):
            return
        ofp = dp.ofproto
        parser = dp.ofproto_parser

        band = parser.OFPMeterBandDrop(rate=int(rate_kbps), burst_size=0)
        req = parser.OFPMeterMod(datapath=dp,
                                 command=ofp.OFPMC_ADD,
                                 flags=ofp.OFPMF_KBPS,
                                 meter_id=meter_id,
                                 bands=[band])
        dp.send_msg(req)
        self.meter_installed.add(key)
        self.logger.info("Created meter dpid=%s meter_id=%s rate=%skbps", dp.id, meter_id, rate_kbps)

