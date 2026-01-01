# small utility to create a random topology at runtime via Python API
from mininet.topo import Topo
from random import randint, choice

class RandomTopo(Topo):
    "Create a random tree-like topology with n hosts and m switches."
    def build(self, num_switches=4, num_hosts=6):
        switches = [self.addSwitch(f's{i+1}') for i in range(num_switches)]
        hosts = [self.addHost(f'h{i+1}') for i in range(num_hosts)]
        # connect hosts randomly to switches
        for h in hosts:
            sw = choice(switches)
            self.addLink(h, sw)
        # chain switches in a random small-world style
        for i in range(len(switches)-1):
            bw = choice([1,5,10])
            self.addLink(switches[i], switches[i+1], bw=bw)
# placeholder: paste full code here
