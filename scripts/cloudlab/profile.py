"""CloudLab profile for a configurable LAN with static node IP addresses."""

import geni.portal as portal
import geni.rspec.pg as pg

pc = portal.Context()
request = pc.makeRequestRSpec()

pc.defineParameter(
    "nodeCount",
    "Number of Nodes",
    portal.ParameterType.INTEGER,
    5,
    longDescription="Number of nodes to allocate in the LAN.",
)

imageList = [
    ("default", "Default Image"),
    ("urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU24-64-STD", "Ubuntu 24.04"),
    ("urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU22-64-STD", "Ubuntu 22.04"),
    ("urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU20-64-STD", "Ubuntu 20.04"),
    ("urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU18-64-STD", "Ubuntu 18.04"),
    ("urn:publicid:IDN+emulab.net+image+emulab-ops//CENTOS9S-64-STD", "CentOS 9 Stream"),
    ("urn:publicid:IDN+emulab.net+image+emulab-ops//CENTOS8S-64-STD", "CentOS 8 Stream"),
    ("urn:publicid:IDN+emulab.net+image+emulab-ops//ROCKY9-64-STD", "Rocky Linux 9"),
    ("urn:publicid:IDN+emulab.net+image+emulab-ops//FBSD135-64-STD", "FreeBSD 13.5"),
    ("urn:publicid:IDN+emulab.net+image+emulab-ops//FBSD142-64-STD", "FreeBSD 14.2"),
]
pc.defineParameter(
    "osImage",
    "Select OS Image",
    portal.ParameterType.IMAGE,
    imageList[0],
    imageList,
    longDescription="OS image for all nodes.",
)

pc.defineParameter(
    "phystype",
    "Optional physical node type",
    portal.ParameterType.NODETYPE,
    "",
    longDescription="Pick one physical node type for all nodes.",
)

pc.defineParameter(
    "linkSpeed",
    "Link Speed",
    portal.ParameterType.INTEGER,
    0,
    [
        (0, "Any"),
        (100000, "100Mb/s"),
        (1000000, "1Gb/s"),
        (10000000, "10Gb/s"),
        (25000000, "25Gb/s"),
        (100000000, "100Gb/s"),
    ],
    longDescription="Requested LAN link speed.",
)

pc.defineParameter(
    "bestEffort",
    "Best Effort",
    portal.ParameterType.BOOLEAN,
    False,
    longDescription="Allow best-effort mapping when bandwidth is constrained.",
)

pc.defineParameter(
    "noInterSwitch",
    "No Interswitch Links",
    portal.ParameterType.BOOLEAN,
    False,
    longDescription="Require all nodes to be on the same switch.",
)

params = pc.bindParameters()

if params.nodeCount < 1:
    pc.reportError(portal.ParameterError("You must choose at least 1 node.", ["nodeCount"]))

pc.verifyParameters()

lan = request.LAN("lan")
if params.bestEffort:
    lan.best_effort = True
elif params.linkSpeed > 0:
    lan.bandwidth = params.linkSpeed

if params.noInterSwitch:
    lan.setNoInterSwitchLinks()

for i in range(params.nodeCount):
    node = request.RawPC("node{}".format(i))
    if params.osImage and params.osImage != "default":
        node.disk_image = params.osImage

    if params.phystype:
        node.hardware_type = params.phystype

    iface = node.addInterface("eth1")
    iface.addAddress(pg.IPv4Address("192.168.1.{}".format(i + 1), "255.255.255.0"))
    lan.addInterface(iface)

pc.printRequestRSpec(request)
