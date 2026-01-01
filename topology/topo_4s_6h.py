#!/usr/bin/env python3
"""
FourSwitchTopo: 4 switches, 6 hosts
- s1 -- s2 -- s3 -- s4    (with a 1 Mbps bottleneck on s2<->s3)
- Hosts:
    h1, h2 -> s1
    h3     -> s2
    h4     -> s3
    h5, h6 -> s4

Exports:
    topos = { 'fourswitch': ( lambda: FourSwitchTopo() ) }

Run example:
    sudo mn --custom topology/topo_4s_6h.py --topo fourswitch \
      --controller=remote --switch ovs,protocols=OpenFlow13
"""

from mininet.topo import Topo
from mininet.link import TCLink


class FourSwitchTopo(Topo):
    """4-switch, 6-host topology with an s2<->s3 bottleneck."""

    def build(self,
              access_bw=10,          # Mbps for normal links
              bottleneck_bw=1,       # Mbps for s2<->s3 bottleneck
              bottleneck_queue=20,   # packets (approx)
              ):
        # Switches (OpenFlow 1.3-capable OVS assumed)
        s1 = self.addSwitch('s1', protocols='OpenFlow13')
        s2 = self.addSwitch('s2', protocols='OpenFlow13')
        s3 = self.addSwitch('s3', protocols='OpenFlow13')
        s4 = self.addSwitch('s4', protocols='OpenFlow13')

        # Hosts
        h1 = self.addHost('h1')
        h2 = self.addHost('h2')
        h3 = self.addHost('h3')
        h4 = self.addHost('h4')
        h5 = self.addHost('h5')
        h6 = self.addHost('h6')

        # Access links (hosts to edge switches)
        self.addLink(h1, s1, cls=TCLink, bw=access_bw)
        self.addLink(h2, s1, cls=TCLink, bw=access_bw)

        self.addLink(h3, s2, cls=TCLink, bw=access_bw)

        self.addLink(h4, s3, cls=TCLink, bw=access_bw)

        self.addLink(h5, s4, cls=TCLink, bw=access_bw)
        self.addLink(h6, s4, cls=TCLink, bw=access_bw)

        # Switch interconnects
        # s1 -- s2 (normal)
        self.addLink(s1, s2, cls=TCLink, bw=access_bw)

        # s2 -- s3 (BOTTLENECK)
        self.addLink(
            s2, s3,
            cls=TCLink,
            bw=bottleneck_bw,
            max_queue_size=bottleneck_queue
        )

        # s3 -- s4 (normal)
        self.addLink(s3, s4, cls=TCLink, bw=access_bw)


# Export topology names for Mininet CLI
topos = {
    'fourswitch': (lambda: FourSwitchTopo()),
    # Optional alias if you prefer a more descriptive name:
    'four_switch_six_host': (lambda: FourSwitchTopo()),
}

if __name__ == '__main__':
    # Optional: quick self-test (not used by Mininet CLI)
    # This block lets you run `python topology/topo_4s_6h.py` safely.
    print("FourSwitchTopo module loaded. Use with Mininet's --custom/--topo.")

