#!/usr/bin/env python3

"""
This is a simple script to compute the Roofline Model
(https://en.wikipedia.org/wiki/Roofline_model) of given HW platforms
running given apps

Peak bandwidth must be specified in GB/s
Peak performance must be specified in GFLOP/s
Arithemtic intensity is specified in FLOP/byte
Performance is specified in GFLOP/s

Copyright 2018, Mohamed A. Bamakhrama
Licensed under BSD license shown in LICENSE
Further edited by Jorge Luis Galvez Vallejo, to only display 
one single roofline and also the flop rate. 

Original can be found under: https://github.com/mohamed/roofline
"""


import csv
import sys
import argparse
import numpy
import matplotlib.pyplot
import matplotlib
#matplotlib.rc('font', family='Times New Roman')


# Constants
# The following constants define the span of the intensity axis
START = -3
STOP = 4
N = abs(STOP - START + 1)


def roofline(num_platforms, peak_performance, peak_bandwidth, intensity):
    """
    Computes the roofline model for the given platforms.
    Returns The achievable performance
    """

    assert isinstance(num_platforms, int) and num_platforms > 0
    assert isinstance(peak_performance, numpy.ndarray)
    assert isinstance(peak_bandwidth, numpy.ndarray)
    assert isinstance(intensity, numpy.ndarray)
    assert (num_platforms == peak_performance.shape[0] and
            num_platforms == peak_bandwidth.shape[0])

    achievable_performance = numpy.zeros((num_platforms, len(intensity)))
    for i in range(num_platforms):
        achievable_performance[i:] = numpy.minimum(peak_performance[i],
                                                   peak_bandwidth[i] * intensity)
    return achievable_performance


def process(hw_platforms, sw_apps, xkcd):
    """
    Processes the hw_platforms and sw_apps to plot the Roofline.
    """
    assert isinstance(hw_platforms, list)
    assert isinstance(sw_apps, list)
    assert isinstance(xkcd, bool)

    # arithmetic intensity
    arithmetic_intensity = numpy.logspace(START, STOP, num=N, base=10)
    # Hardware platforms
    platforms = [p[0] for p in hw_platforms]

    # Compute the rooflines
    achievable_performance = roofline(len(platforms),
                                      numpy.array([p[1] for p in hw_platforms]),
                                      numpy.array([p[2] for p in hw_platforms]),
                                      arithmetic_intensity)

    # Apps
    if sw_apps != []:
        apps = [a[0] for a in sw_apps]
        apps_intensity = numpy.array([a[1] for a in sw_apps])
        floprate = numpy.array([a[2] for a in sw_apps])

    # Plot the graphs
    if xkcd:
        matplotlib.pyplot.xkcd()
    fig, axis = matplotlib.pyplot.subplots()
    axis.set_xscale('log', basex=10)
    axis.set_yscale('log', basey=10)
    axis.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=14)
    axis.grid(True, which='both', color='gray', linestyle='-',linewidth=0.2)

    #matplotlib.pyplot.setp(axis, xticks=numpy.logspace(1,20,num=20,base=10),
    matplotlib.pyplot.setp(axis, xticks=arithmetic_intensity,
                           yticks=numpy.logspace(1, 20, num=20, base=10))

    matplotlib.pyplot.yticks(fontsize=12)
    matplotlib.pyplot.xticks(fontsize=12)
    axis.set_ylabel("FLOP-rate (GFLOP/s)", fontsize=14)
    axis.set_ylim(bottom=10, top=100500)

    #axis.set_title('Roofline Plot', fontsize=14)

    l1 = numpy.array((0.12,100))
    l2 = numpy.array((10,35))
    trans_angle = matplotlib.pyplot.gca().transData.transform_angles(numpy.array((75,)), l2.reshape((1,2)))[0]
    th1 = matplotlib.pyplot.text(l1[0],l1[1], ' HBM BW: 778 GB/s ', fontsize=10, rotation=trans_angle,rotation_mode='anchor')
    th2 = matplotlib.pyplot.text(20,10000, ' DP Peak: 7.66 TF/s ', fontsize=10)
    for idx, val in enumerate(platforms):
        axis.plot(arithmetic_intensity, achievable_performance[idx, 0:],
                     label=val)

    if sw_apps != []:
        color = matplotlib.pyplot.cm.rainbow(numpy.linspace(0, 1, len(apps)))
        for idx, val in enumerate(apps):
            axis.plot(apps_intensity[idx], floprate[idx], label=val,
                         linestyle='-.', marker='o', color=color[idx])

    axis.legend(loc='upper left', prop={'size': 9})
    fig.tight_layout()
    #matplotlib.pyplot.show()
    matplotlib.pyplot.savefig('plot_roofline.png', dpi=500 )


def read_file(filename, row_len, csv_name):
    """
    Reads CSV file and returns a list of row_len-ary tuples
    """
    assert isinstance(row_len, int)
    elements = list()
    try:
        in_file = open(filename, 'r') if filename is not None else sys.stdin
        reader = csv.reader(in_file, dialect='excel')
        for row in reader:
            if len(row) != row_len:
                print("Error: Each row in %s must be contain exactly %d entries!"
                      % (csv_name, row_len), file=sys.stderr)
                sys.exit(1)
            element = tuple([row[0]] + [float(r) for r in row[1:]])
            elements.append(element)
        if filename is not None:
            in_file.close()
    except IOError as ex:
        print(ex, file=sys.stderr)
        sys.exit(1)
    return elements


def main():
    """
    main function
    """
    hw_platforms = list()
    apps = list()
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", metavar="hw_csv", help="HW platforms CSV file", type=str)
    parser.add_argument("-a", metavar="apps_csv", help="applications CSV file", type=str)
    parser.add_argument("--hw-only", action='store_true', default=False)
    parser.add_argument("--xkcd", action='store_true', default=False)

    args = parser.parse_args()
    # HW
    print("Reading HW characteristics...")
    hw_platforms = read_file(args.i, 4, "HW CSV")
    # apps
    if args.hw_only:
        print("Plotting only HW characteristics without any applications...")
        apps = list()
    else:
        print("Reading applications intensities...")
        apps = read_file(args.a, 3, "SW CSV")

    print(hw_platforms)
    print("Plotting using XKCD plot style is set to %s" % (args.xkcd))
    if apps != []:
        print(apps)
    process(hw_platforms, apps, args.xkcd)
    sys.exit(0)


if __name__ == "__main__":
    main()


'''
BSD 3-Clause License

Copyright (c) 2018, Mohamed Bamakhrama
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
