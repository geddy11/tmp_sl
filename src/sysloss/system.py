# MIT License
#
# Copyright (c) 2024, Geir Drange
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import rustworkx as rx
import numpy as np
from rich.tree import Tree
import json
import pandas as pd
from sysloss.components import *
from sysloss.components import (
    ComponentTypes,
    _get_opt,
    _get_mand,
    RS_DEFAULT,
    LIMITS_DEFAULT,
)


class System:
    """System to be analyzed."""

    def __init__(self, name: str, source):
        self.g = None
        if not isinstance(source, Source):
            raise ValueError("Error: First component of system must be a source!")

        self.g = rx.PyDAG(check_cycle=True, multigraph=False, attrs={})
        pidx = self.g.add_node(source)
        self.g.attrs["name"] = name
        self.g.attrs["nodes"] = {}
        self.g.attrs["nodes"][source.params["name"]] = pidx

    @classmethod
    def from_file(cls, fname: str):
        """Load system from json file"""
        with open(fname, "r") as f:
            sys = json.load(f)

        # add root
        name = _get_mand(sys["root"]["params"], "name")
        vo = _get_mand(sys["root"]["params"], "vo")
        rs = _get_opt(sys["root"]["params"], "rs", RS_DEFAULT)
        lim = _get_opt(sys["root"], "limits", LIMITS_DEFAULT)
        self = cls(sys["name"], Source(name, vo=vo, rs=rs, limits=lim))
        # add childs
        if sys["childs"] != {}:
            for p in sys["childs"]:
                # print(p)
                for c in sys["childs"][p]:
                    cname = _get_mand(c["params"], "name")
                    # print("  " + cname)
                    limits = _get_opt(c, "limits", LIMITS_DEFAULT)
                    iq = _get_opt(c["params"], "iq", 0.0)
                    if c["type"] == "CONVERTER":
                        vo = _get_mand(c["params"], "vo")
                        eff = _get_mand(c["params"], "eff")
                        self.add_comp(
                            p,
                            comp=Converter(cname, vo=vo, eff=eff, iq=iq, limits=limits),
                        )
                    elif c["type"] == "LINREG":
                        vo = _get_mand(c["params"], "vo")
                        vdrop = _get_opt(c["params"], "vdrop", 0.0)
                        self.add_comp(
                            p,
                            comp=LinReg(
                                cname, vo=vo, vdrop=vdrop, iq=iq, limits=limits
                            ),
                        )
                    elif c["type"] == "LOSS":
                        rs = _get_mand(c["params"], "rs")
                        vdrop = _get_mand(c["params"], "vdrop")
                        self.add_comp(
                            p, comp=Loss(cname, rs=rs, vdrop=vdrop, limits=limits)
                        )
                    elif c["type"] == "LOAD":
                        if "pwr" in c["params"]:
                            pwr = _get_mand(c["params"], "pwr")
                            self.add_comp(p, comp=PLoad(cname, pwr=pwr, limits=limits))
                        elif "rs" in c["params"]:
                            rs = _get_mand(c["params"], "rs")
                            self.add_comp(p, comp=RLoad(cname, rs=rs, limits=limits))
                        else:
                            ii = _get_mand(c["params"], "ii")
                            self.add_comp(p, comp=ILoad(cname, ii=ii, limits=limits))

        return self

    def __get_index(self, name: str):
        """Get node index from component name"""
        if name in self.g.attrs["nodes"]:
            return self.g.attrs["nodes"][name]

        return -1

    def __chk_parent(self, parent: str):
        """Check if parent exists"""
        if not parent in self.g.attrs["nodes"].keys():
            raise ValueError('Error: Parent name "{}" not found!'.format(parent))

        return True

    def __chk_name(self, name: str):
        """Check if component name is valid"""
        # check if name exists
        if name in self.g.attrs["nodes"].keys():
            raise ValueError('Error: Component name "{}" is already used!'.format(name))

        return True

    def __get_childs_tree(self):
        """Get dict of parent/childs"""
        # if rev == True:
        #    childs = list(reversed(rx.bfs_successors(self.g, 0)))
        # else:
        childs = list(rx.bfs_successors(self.g, 0))
        cdict = {}
        for c in childs:
            cs = []
            for l in c[1]:
                cs += [self.g.attrs["nodes"][l.params["name"]]]
            cdict[self.g.attrs["nodes"][c[0].params["name"]]] = cs
        return cdict

    def __get_nodes(self):
        """Get list of nodes in system"""
        return [n for n in self.g.node_indices()]

    def __get_childs(self):
        """Get list of children of each node"""
        nodes = self.__get_nodes()
        cs = list(-np.ones(max(nodes) + 1, dtype=np.int32))
        for n in nodes:
            if self.g.out_degree(n) > 0:
                ind = [i for i in self.g.successor_indices(n)]
                cs[n] = ind
        return cs

    def __get_parents(self):
        """Get list of parent of each node"""
        nodes = self.__get_nodes()
        ps = list(-np.ones(max(nodes) + 1, dtype=np.int32))
        for n in nodes:
            if self.g.in_degree(n) > 0:
                ind = [i for i in self.g.predecessor_indices(n)]
                ps[n] = ind
        return ps

    def __get_sources(self):
        """Get list of sources"""
        tn = [n for n in rx.topological_sort(self.g)]
        return [n for n in tn if isinstance(self.g[n], Source)]

    def __get_topo_sort(self):
        """Get nodes topological sorted"""
        tps = rx.topological_sort(self.g)
        return [n for n in tps]

    def __sys_vars(self):
        """Get system variable lists"""
        vn = max(self.__get_nodes()) + 1  # highest node index + 1
        v = list(np.zeros(vn))  # voltages
        i = list(np.zeros(vn))  # currents
        return v, i

    def __make_rtree(self, adj, node):
        """Create Rich tree"""
        tree = Tree(node)
        for child in adj.get(node, []):
            tree.add(self.__make_rtree(adj, child))
        return tree

    def add_comp(self, parent: str, *, comp):
        """Add component to system"""
        # check that parent exists
        self.__chk_parent(parent)
        # check that component name is unique
        self.__chk_name(comp.params["name"])

        pidx = self.__get_index(parent)

        # check that parent allows component type as child
        if not comp.component_type in self.g[pidx].child_types:
            raise ValueError(
                "Error: Parent does not allow child of type {}!".format(
                    comp.component_type.name
                )
            )
        cidx = self.g.add_child(pidx, comp, None)
        self.g.attrs["nodes"][comp.params["name"]] = cidx

    def add_source(self, source):
        """Add new source"""
        self.__chk_name(source.params["name"])
        if not isinstance(source, Source):
            raise ValueError("Error: Component must be a source!")

        pidx = self.g.add_node(source)
        self.g.attrs["nodes"][source.params["name"]] = pidx

    def change_comp(self, name: str, *, comp):
        """Replace component with a new one"""
        # if component name changes, check that it is unique
        if name != comp.params["name"]:
            self.__chk_name(comp.params["name"])

        eidx = self.__get_index(name)
        # check that parent allows component type as child
        parents = self.__get_parents()
        if parents[eidx] != -1:
            if not comp.component_type in self.g[parents[eidx][0]].child_types:
                raise ValueError(
                    "Error: Parent does not allow child of type {}!".format(
                        comp.component_type.name
                    )
                )
        self.g[eidx] = comp
        # replace node name in graph dict
        del [self.g.attrs["nodes"][name]]
        self.g.attrs["nodes"][comp.params["name"]] = eidx

    def del_comp(self, name: str, *, del_childs: bool = True):
        eidx = self.__get_index(name)
        if eidx == -1:
            raise ValueError("Error: Component name does not exist!")
        parents = self.__get_parents()
        if parents[eidx] == -1:
            raise ValueError("Error: Cannot delete source node!")
        childs = self.__get_childs()
        # if not leaf, check if child type is allowed by parent type (not possible?)
        # if leaves[eidx] == 0:
        #     for c in childs[eidx]:
        #         if not self.g[c].component_type in self.g[parents[eidx]].child_types:
        #             raise ValueError(
        #                 "Error: Parent and child of component are not compatible!"
        #             )
        # delete childs first if selected
        if del_childs:
            for c in rx.descendants(self.g, eidx):
                print(c, eidx)
                del [self.g.attrs["nodes"][self.g[c].params["name"]]]
                self.g.remove_node(c)
        # delete node
        self.g.remove_node(eidx)
        del [self.g.attrs["nodes"][name]]
        # restore links between new parent and childs, unless deleted
        if not del_childs:
            if childs[eidx] != -1:
                for c in childs[eidx]:
                    self.g.add_edge(parents[eidx][0], c, None)

    def tree(self, name=""):
        """Print tree structure, starting from node name"""
        if not name == "":
            if not name in self.g.attrs.keys():
                raise ValueError("Error: Component name is not valid!")
            root = [name]
        else:
            ridx = self.__get_sources()
            root = [self.g[n].params["name"] for n in ridx]

        t = Tree(self.g.attrs["name"])
        for n in root:
            adj = rx.bfs_successors(self.g, self.g.attrs["nodes"][n])
            ndict = {}
            for i in adj:
                c = []
                for j in i[1]:
                    c += [j.params["name"]]
                ndict[i[0].params["name"]] = c
            t.add(self.__make_rtree(ndict, n))
        return t

    def __sys_init(self):
        """Create vectors of init values for solver"""
        v, i = self.__sys_vars()
        for n in self.__get_nodes():
            v[n] = self.g[n]._get_outp_voltage()
            i[n] = self.g[n]._get_inp_current()
        return v, i

    def __fwd_prop(self, v: float, i: float):
        """Forward propagation of voltages"""
        vo, _ = self.__sys_vars()
        # update output voltages (per node)
        for n in self.__topo_nodes:
            p = self.__parents[n]
            if self.__childs[n] == -1:  # leaf
                if p == -1:  # root
                    vo[n] = self.g[n]._solv_outp_volt(0.0, 0.0, 0.0)
                else:
                    vo[n] = self.g[n]._solv_outp_volt(v[p[0]], i[n], 0.0)
            else:
                # add currents into childs
                isum = 0
                for c in self.__childs[n]:
                    isum += i[c]
                if p == -1:  # root
                    vo[n] = self.g[n]._solv_outp_volt(0.0, 0.0, isum)
                else:
                    vo[n] = self.g[n]._solv_outp_volt(v[p[0]], i[n], isum)
        return vo

    def __back_prop(self, v: float, i: float):
        """Backward propagation of currents"""
        _, ii = self.__sys_vars()
        # update input currents (per node)
        for n in self.__topo_nodes[::-1]:
            p = self.__parents[n]
            if self.__childs[n] == -1:  # leaf
                if p == -1:  # root
                    ii[n] = self.g[n]._solv_inp_curr(v[n], 0.0, 0.0)
                else:
                    ii[n] = self.g[n]._solv_inp_curr(v[p[0]], 0.0, 0.0)
            else:
                isum = 0.0
                for c in self.__childs[n]:
                    isum += i[c]
                if p == -1:  # root
                    ii[n] = self.g[n]._solv_inp_curr(v[n], v[n], isum)
                else:
                    ii[n] = self.g[n]._solv_inp_curr(v[p[0]], v[n], isum)

        return ii

    def __rel_update(self):
        """Update lists with component relationships"""
        self.__parents = self.__get_parents()
        self.__childs = self.__get_childs()
        self.__topo_nodes = self.__get_topo_sort()

    def __get_parent_name(self, node):
        """Get parent name of node"""
        if self.__parents[node] == -1:
            return ""

        return self.g[self.__parents[node][0]].params["name"]

    def solve(self, *, vtol=1e-5, itol=1e-6, maxiter=1000, quiet=True):
        """Analyze system"""
        self.__rel_update()
        # initial condition
        v, i = self.__sys_init()
        # solve system function
        iters = 1
        while iters <= maxiter:
            vi = self.__fwd_prop(v, i)
            ii = self.__back_prop(vi, i)
            if np.allclose(np.array(v), np.array(vi), rtol=vtol) and np.allclose(
                np.array(i), np.array(ii), rtol=itol
            ):
                if not quiet:
                    print("Tolerances met after {} iterations".format(iters))
                break
            v, i = vi, ii
            iters += 1

        if iters > maxiter:
            print("Analysis aborted after {} iterations".format(iters - 1))
            return None

        # calculate results for each node
        names, parent, typ, pwr, loss = [], [], [], [], []
        eff, warn, vsi, iso = [], [], [], []
        domain, dname = [], "none"
        for n in self.__topo_nodes:  # [vi, vo, ii, io]
            names += [self.g[n].params["name"]]
            if self.g[n].component_type.name == "SOURCE":
                dname = self.g[n].params["name"]
            domain += [dname]
            vi = v[n]
            vo = v[n]
            ii = i[n]
            io = i[n]
            p = self.__parents[n]

            if p == -1:  # root
                vi = v[n] + self.g[n].params["rs"] * ii
            elif self.__childs[n] == -1:  # leaf
                vi = v[p[0]]
                io = 0.0
            else:
                io = 0.0
                for c in self.__childs[n]:
                    io += i[c]
                vi = v[p[0]]
            parent += [self.__get_parent_name(n)]
            p, l, e = self.g[n]._solv_pwr_loss(vi, vo, ii, io)
            pwr += [p]
            loss += [l]
            eff += [e]
            typ += [self.g[n].component_type.name]
            warn += [self.g[n]._solv_get_warns(vi, vo, ii, io)]
            vsi += [vi]
            iso += [io]

        # remove unused node indices
        vso, isi = [], []
        for n in self.__topo_nodes:
            if n in self.__get_nodes():
                vso += [v[n]]
                isi += [i[n]]

        # report
        res = {}
        res["Component"] = names
        res["Type"] = typ
        res["Parent"] = parent
        res["Domain"] = domain
        res["Vin (V)"] = vsi
        res["Vout (V)"] = vso
        res["Iin (A)"] = isi
        res["Iout (A)"] = iso
        res["Power (W)"] = pwr  # [res['Power (W)'].sum() - res['Power (W)'][0]]
        res["Loss (W)"] = loss  # [res['Loss (W)'].sum()]
        res["Efficiency (%)"] = eff
        res["Warnings"] = warn
        df = pd.DataFrame(res)
        tpwr = abs(vsi[0] * i[0])
        tloss = df["Loss (W)"].sum()
        w = "None"
        if sum(df["Warnings"] != "") > 0:
            w = "Yes"
        eff = 0.0
        if tpwr > 0.0:
            eff = (tpwr - tloss) / tpwr
        df.loc[len(df)] = [
            "System total",
            "",
            "",
            "",
            "",
            "",
            i[0],
            "",
            tpwr,
            tloss,
            eff,
            w,
        ]

        return df

    def params(self, limits=False):
        """Return component parameters"""
        self.__rel_update()
        names, typ, parent, vo, vdrop = [], [], [], [], []
        iq, rs, eff, ii, pwr = [], [], [], [], []
        lii, lio, lvi, lvo = [], [], [], []
        domain, dname = [], "none"
        for n in self.__topo_nodes:
            names += [self.g[n].params["name"]]
            typ += [self.g[n].component_type.name]
            if self.g[n].component_type.name == "SOURCE":
                dname = self.g[n].params["name"]
            domain += [dname]
            _vo, _vdrop, _iq, _rs, _eff, _ii, _pwr = "", "", "", "", "", "", ""
            if self.g[n].component_type == ComponentTypes.SOURCE:
                _vo = self.g[n].params["vo"]
                _rs = self.g[n].params["rs"]
            elif self.g[n].component_type == ComponentTypes.LOAD:
                if "pwr" in self.g[n].params:
                    _pwr = self.g[n].params["pwr"]
                elif "rs" in self.g[n].params:
                    _rs = self.g[n].params["rs"]
                else:
                    _ii = self.g[n].params["ii"]
            elif self.g[n].component_type == ComponentTypes.CONVERTER:
                _vo = self.g[n].params["vo"]
                _iq = self.g[n].params["iq"]
                _eff = self.g[n].params["eff"]
            elif self.g[n].component_type == ComponentTypes.LINREG:
                _vo = self.g[n].params["vo"]
                _vdrop = self.g[n].params["vdrop"]
                _iq = self.g[n].params["iq"]
            elif self.g[n].component_type == ComponentTypes.LOSS:
                _vdrop = self.g[n].params["vdrop"]
                _rs = self.g[n].params["rs"]
            vo += [_vo]
            vdrop += [_vdrop]
            iq += [_iq]
            rs += [_rs]
            eff += [_eff]
            ii += [_ii]
            pwr += [_pwr]
            parent += [self.__get_parent_name(n)]
            if limits:
                lii += [self.g[n].limits["ii"]]
                lio += [self.g[n].limits["io"]]
                lvi += [self.g[n].limits["vi"]]
                lvo += [self.g[n].limits["vo"]]
        # report
        res = {}
        res["Component"] = names
        res["Type"] = typ
        res["Parent"] = parent
        res["Domain"] = domain
        res["vo (V)"] = vo
        res["vdrop (V)"] = vdrop
        res["iq (A)"] = iq
        res["rs (Ohm)"] = rs
        res["eff (%)"] = eff
        res["ii (A)"] = ii
        res["pwr (W)"] = pwr
        if limits:
            res["vi limits (V)"] = lvi
            res["vo limits (V)"] = lvo
            res["ii limits (A)"] = lii
            res["io limits (A)"] = lio
        return pd.DataFrame(res)

    def save(self, fname, *, indent=4):
        """Save system as json file"""
        self.__rel_update()
        sys = {
            "name": self.g.attrs["name"],
            "root": {
                "type": self.g[0].component_type.name,
                "params": self.g[0].params,
                "limits": self.g[0].limits,
            },
        }
        tree = self.__get_childs_tree()
        cdict = {}
        if tree != {}:
            for e in tree:
                childs = []
                for c in tree[e]:
                    childs += [
                        {
                            "type": self.g[c].component_type.name,
                            "params": self.g[c].params,
                            "limits": self.g[c].limits,
                        }
                    ]
                cdict[self.g[e].params["name"]] = childs
        sys["childs"] = cdict

        with open(fname, "w") as f:
            json.dump(sys, f, indent=indent)
