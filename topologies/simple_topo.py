"""
Simple Mininet topology for quick testing with Ryu.

Usage (on a Linux machine with Mininet installed):
  sudo python topologies/simple_topo.py --controller-ip 127.0.0.1 --controller-port 6633

Or use Mininet CLI to launch the custom topo:
  sudo mn --custom topologies/simple_topo.py --topo simple --controller=remote,ip=127.0.0.1,port=6633

Topology layout:
    h1   h2
     \   /
      s1
      |
      s2
     / \
   h3   h4

This script starts a RemoteController pointing to `--controller-ip` and drops you into the Mininet CLI.
"""
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel
import argparse


class SimpleTopo(Topo):
    def build(self):
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        h1 = self.addHost('h1')
        h2 = self.addHost('h2')
        h3 = self.addHost('h3')
        h4 = self.addHost('h4')

        # connect hosts to switches
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        # inter-switch link
        self.addLink(s1, s2)
        # hosts on s2
        self.addLink(h3, s2)
        self.addLink(h4, s2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller-ip', type=str, default='127.0.0.1', help='Remote controller IP (Ryu)')
    parser.add_argument('--controller-port', type=int, default=6633, help='Remote controller port')
    args = parser.parse_args()

    setLogLevel('info')
    topo = SimpleTopo()
    controller = RemoteController('c0', ip=args.controller_ip, port=args.controller_port)
    net = Mininet(topo=topo, controller=controller, link=TCLink)
    net.start()
    print('\nMininet started. Controller set to %s:%s' % (args.controller_ip, args.controller_port))
    print('Try: pingall, h1 ping h3, iperf tests from hosts, or run your Ryu app.')
    CLI(net)
    net.stop()


if __name__ == '__main__':
    main()
