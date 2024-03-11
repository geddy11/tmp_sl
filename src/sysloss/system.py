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
from sysloss.elements import *
from sysloss.elements import (
    ElementTypes,
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
            raise ValueError("Error: First element of system must be a source!")
        else:
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
                        self.add_element(
                            p,
                            element=Converter(
                                cname, vo=vo, eff=eff, iq=iq, limits=limits
                            ),
                        )
                    elif c["type"] == "LINREG":
                        vo = _get_mand(c["params"], "vo")
                        vdrop = _get_opt(c["params"], "vdrop", 0.0)
                        self.add_element(
                            p,
                            element=LinReg(
                                cname, vo=vo, vdrop=vdrop, iq=iq, limits=limits
                            ),
                        )
                    elif c["type"] == "LOSS":
                        rs = _get_mand(c["params"], "rs")
                        vdrop = _get_mand(c["params"], "vdrop")
                        self.add_element(
                            p, element=Loss(cname, rs=rs, vdrop=vdrop, limits=limits)
                        )
                    elif c["type"] == "LOAD":
                        if "pwr" in c["params"]:
                            pwr = _get_mand(c["params"], "pwr")
                            self.add_element(
                                p, element=PLoad(cname, pwr=pwr, limits=limits)
                            )
                        else:
                            ii = _get_mand(c["params"], "ii")
                            self.add_element(
                                p, element=ILoad(cname, ii=ii, limits=limits)
                            )

        return self

    def __get_index(self, name: str):
        """Get node index from element name"""
        if name in self.g.attrs["nodes"]:
            return self.g.attrs["nodes"][name]

        return -1

    def __chk_parent(self, parent: str):
        """Check if parent exists"""
        if type(parent) != str:
            raise ValueError("Error: Parent name must be a string!")
        # check if parent exists
        if not parent in self.g.attrs["nodes"].keys():
            raise ValueError('Error: Parent name "{}" not found!'.format(parent))
            return False

        return True

    # check if element name is valid
    def __chk_name(self, name: str):
        """Check if element name is valid"""
        # check name type
        if type(name) != str:
            raise ValueError("Error: Element name must be a string!")
            return False
        # check if exists exists
        pidx = self.__get_index(name)
        if name in self.g.attrs["nodes"].keys():
            raise ValueError('Error: Element name "{}" is already used!'.format(name))
            return False

        return True

    def __get_childs(self, rev: bool = True):
        """Get dict of parent/childs"""
        if rev == True:
            childs = list(reversed(rx.bfs_successors(self.g, 0)))
        else:
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
        nodes = []
        for n in self.g.nodes():
            nodes += [self.g.attrs["nodes"][n.params["name"]]]
        return sorted(nodes)

    def __get_parents(self):
        """Get list of parent of each child"""
        nodes = self.__get_nodes()
        ps = np.zeros(max(nodes) + 1, dtype=np.int32)
        for n in nodes:
            if n != 0:
                nodename = [
                    i for i in self.g.attrs["nodes"] if self.g.attrs["nodes"][i] == n
                ]
                # pname = [self.__get_parent_name(self.g.attrs['nodes'][n])]
                pname = self.g.predecessors(self.g.attrs["nodes"][nodename[0]])[
                    0
                ].params["name"]
                pidx = self.g.attrs["nodes"][pname]
                ps[n] = self.g.attrs["nodes"][
                    self.g.predecessors(self.g.attrs["nodes"][nodename[0]])[0].params[
                        "name"
                    ]
                ]
                # print("n={}, pname={}, ps[n]={}, pidx={}".format(n, nodename, pname, ps[n], pidx))
        return list(ps)

    def __get_edges(self):
        """Get list of element connections (edges)"""
        return list(reversed(rx.dfs_edges(self.g, 0)))

    def __get_leaves(self):
        """Get list of leaf nodes"""
        nodes = self.__get_nodes()
        ls = np.zeros(max(nodes) + 1, dtype=np.int32)
        for n in nodes:
            if self.g.out_degree(n) == 0:
                ls[n] = 1
        return list(ls)

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

    def add_element(self, parent: str, *, element):
        """Add element to system"""
        # check that parent exists
        if not self.__chk_parent(parent):
            raise ValueError("Error: Parent name does not exist!")
            return

        # check that element name is unique
        if not self.__chk_name(element.params["name"]):
            raise ValueError("Error: Element name already taken!")
            return

        pidx = self.__get_index(parent)

        # check that parent allows element type as child
        if not element.element_type in self.g[pidx].child_types:
            raise ValueError(
                "Error: Parent does not allow child of type {}!".format(
                    element.element_type.name
                )
            )
            return

        cidx = self.g.add_child(pidx, element, None)
        # print('Add {} to {}'.format(cidx, pidx))
        self.g.attrs["nodes"][element.params["name"]] = cidx

    def change_element(self, *, name: str, element):
        """Replace element with a new one"""
        # if element name changes, check that it is unique
        if name != element.params["name"]:
            if not self.__chk_name(element.params["name"]):
                raise ValueError("Error: Element name already taken!")
                return

        eidx = self.__get_index(name)
        # check that parent allows element type as child
        if eidx != 0:
            parents = self.__get_parents()
            if not element.element_type in self.g[parents[eidx]].child_types:
                raise ValueError(
                    "Error: Parent does not allow child of type {}!".format(
                        element.element_type.name
                    )
                )
                return

        self.g[eidx] = element
        # replace node name in graph dict
        del [self.g.attrs["nodes"][name]]
        self.g.attrs["nodes"][element.params["name"]] = eidx

    def del_element(self, *, name: str, del_childs: bool = True):
        eidx = self.__get_index(name)
        if eidx == -1:
            raise ValueError("Error: Element name does not exist!")
        if eidx == 0:
            raise ValueError("Error: Cannot delete source node!")
        parents = self.__get_parents()
        leaves = self.__get_leaves()
        childs = self.__get_childs(rev=True)
        # if not leaf, check if child type is allowed by parent type
        if leaves[eidx] == 0:
            # print(childs[eidx], parents[eidx])
            for c in childs[eidx]:
                if not self.g[c].element_type in self.g[parents[eidx]].child_types:
                    raise ValueError(
                        "Error: Parent and child of element are not compatible!"
                    )
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
            if leaves[eidx] == 0:
                for c in childs[eidx]:
                    self.g.add_edge(parents[eidx], c, None)

    def tree(self, name=""):
        """Print tree structure, starting from node name"""
        if not name == "":
            if not name in self.g.attrs.keys():
                raise ValueError("Error: Element name is not valid!")
        else:
            root = self.g[0].params["name"]

        adj = rx.bfs_successors(self.g, self.g.attrs["nodes"][root])
        ndict = {}
        for i in adj:
            c = []
            for j in i[1]:
                c += [j.params["name"]]
            ndict[i[0].params["name"]] = c
        return self.__make_rtree(ndict, root)

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
        for n in self.__nodes:
            if self.__leaves[n] == 1:  # leaf
                if n == 0:  # root
                    vo[n] = self.g[n]._solv_outp_volt(0.0, 0.0, 0.0)
                    # print('leaf, root:', vo[n], n)
                else:
                    p = self.__parents[n]
                    vo[n] = self.g[n]._solv_outp_volt(v[p], i[p], 0.0)
                    # print('leaf: ', vo[n], n)
            else:
                # add currents into childs
                isum = 0
                for c in self.__childs_f[n]:
                    isum += i[c]
                if n == 0:  # root
                    vo[n] = self.g[n]._solv_outp_volt(0.0, 0.0, isum)
                    # print('root:', vo[n], n)
                else:
                    p = self.__parents[n]
                    vo[n] = self.g[n]._solv_outp_volt(v[p], i[p], isum)
                    # print('element:' , vo[n], n)

        return vo

    def __back_prop(self, v: float, i: float):
        """Backward propagation of currents"""
        _, io = self.__sys_vars()
        # update input currents (per edge)
        for e in self.__edges:
            if self.__leaves[e[1]] == 1:  # leaf
                io[e[1]] = self.g[e[1]]._solv_inp_curr(v[e[0]], 0.0, 0.0)
            else:
                c = self.__childs_b[e[1]]
                io[e[1]] = self.g[e[1]]._solv_inp_curr(v[e[0]], v[e[1]], i[c[0]])
        # add currents into childs from root
        if self.__childs_b != {}:
            for c in self.__childs_b[0]:
                io[0] += i[c]

        return io

    def __rel_update(self):
        """Update lists with element relationships"""
        self.__nodes = self.__get_nodes()
        self.__edges = self.__get_edges()
        self.__childs_f = self.__get_childs(rev=False)
        self.__childs_b = self.__get_childs(rev=True)
        self.__parents = self.__get_parents()
        self.__leaves = self.__get_leaves()

    def __get_parent_name(self, node):
        if node == 0:
            return ""

        return self.g[self.__parents[node]].params["name"]

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
        names, parent, typ, pwr, loss, eff, warn, vsi, iso = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for n in self.__nodes:  # [vi, vo, ii, io]
            names += [self.g[n].params["name"]]
            vi = v[n]
            vo = v[n]
            ii = i[n]
            io = i[n]

            if n == 0:  # root
                # parent += ['']
                # print(n, v[n], v[n], i[n], i[n], p, l)
                vi = v[n] + self.g[n].params["rs"] * ii
            elif self.__leaves[n] == 1:  # leaf
                vi = v[self.__parents[n]]
                io = 0.0
                # parent += [self.g[self.__parents[n]].params['name']]
                # print(n, v[n], 0.0, i[n], 0.0, p, l)
            else:
                io = 0.0
                for c in self.__childs_f[n]:
                    io += i[c]
                vi = v[self.__parents[n]]
                # parent += [self.g[self.__parents[n]].params['name']]
                # print(n, v[self.__parents[n]], v[n], i[n], isum, p, l)

            parent += [self.__get_parent_name(n)]
            p, l, e = self.g[n]._solv_pwr_loss(vi, vo, ii, io)
            pwr += [p]
            loss += [l]
            eff += [e]
            typ += [self.g[n].element_type.name]
            warn += [self.g[n]._solv_get_warns(vi, vo, ii, io)]
            vsi += [vi]
            iso += [io]

        # remove unused node indices
        vso, isi = [], []
        for n in range(len(v)):
            if n in self.__nodes:
                vso += [v[n]]
                isi += [i[n]]

        # report
        res = {}
        res["Element"] = names
        res["Type"] = typ
        res["Parent"] = parent
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
            vsi[0],
            0.0,
            i[0],
            0.0,
            tpwr,
            tloss,
            eff,
            w,
        ]

        return df

    def params(self, limits=False):
        """Return element parameters"""
        self.__rel_update()
        # print(self.__parents)
        # print(self.__childs_f)
        # extract params

        names, typ, parent, vo, vdrop, iq, rs, eff, ii, pwr = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        lii, lio, lvi, lvo = [], [], [], []
        for n in self.__nodes:
            names += [self.g[n].params["name"]]
            typ += [self.g[n].element_type.name]
            _vo, _vdrop, _iq, _rs, _eff, _ii, _pwr = "", "", "", "", "", "", ""
            if self.g[n].element_type == ElementTypes.SOURCE:
                _vo = self.g[n].params["vo"]
                _rs = self.g[n].params["rs"]
            elif self.g[n].element_type == ElementTypes.LOAD:
                if "pwr" in self.g[n].params:
                    _pwr = self.g[n].params["pwr"]
                else:
                    _ii = self.g[n].params["ii"]
            elif self.g[n].element_type == ElementTypes.CONVERTER:
                _vo = self.g[n].params["vo"]
                _iq = self.g[n].params["iq"]
                _eff = self.g[n].params["eff"]
            elif self.g[n].element_type == ElementTypes.LINREG:
                _vo = self.g[n].params["vo"]
                _vdrop = self.g[n].params["vdrop"]
                _iq = self.g[n].params["iq"]
            elif self.g[n].element_type == ElementTypes.LOSS:
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
        res["Element"] = names
        res["Type"] = typ
        res["Parent"] = parent
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
                "type": self.g[0].element_type.name,
                "params": self.g[0].params,
                "limits": self.g[0].limits,
            },
        }
        tree = self.__get_childs(rev=False)
        cdict = {}
        if tree != {}:
            for e in tree:
                # print(self.g[e].params['name'], tree[e])
                childs = []
                for c in tree[e]:
                    childs += [
                        {
                            "type": self.g[c].element_type.name,
                            "params": self.g[c].params,
                            "limits": self.g[c].limits,
                        }
                    ]
                cdict[self.g[e].params["name"]] = childs
                # cdict[]
        sys["childs"] = cdict

        with open(fname, "w") as f:
            json.dump(sys, f, indent=indent)
