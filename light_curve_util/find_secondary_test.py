from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from light_curve_util import find_secondary as fs
from astronet.data import preprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys


def match_tce(tic, input_file):
    tce_table = pd.read_csv(input_file, header=0, usecols=[0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 18, 19],
                            dtype={'Sectors':
                                                                                                              int,
                                                                                                        'camera': int,
                                                                                                        'ccd': int})

    match = tce_table[tce_table['tic_id'] == tic]
    if len(match) == 0:
        print('TCE not found')
        return
    else:
        return match


def test_secondary(tic, input_file):
    match = match_tce(tic, input_file)
    sector = match['Sectors'].values[0]

    t0 = match['Epoc'].values[0]
    period = match['Period'].values[0]
    duration = match['Duration'].values[0] / 24.
    Qingress = min([max([match['Qingress'].values[0], 0]), 0.4])

    time, flux, flux_small, flux_big = preprocess.read_and_process_light_curve(tic, tess_data_dir,
                                                                    sector, injected=False)

    zero = t0 - period / 2
    transit_index = np.floor((time - zero) / period)
    odd = np.where(transit_index % 2 == 0)
    even = np.where(transit_index % 2 == 1)

    odd_time, odd_flux = preprocess.phase_fold_and_sort_light_curve(time[odd], flux[odd], period, t0)
    even_time, even_flux = preprocess.phase_fold_and_sort_light_curve(time[even], flux[even], period, t0)

    odd_depth = preprocess.measure_eclipse_depth(odd_time, odd_flux, duration, Qingress)
    even_depth = preprocess.measure_eclipse_depth(even_time, even_flux, duration, Qingress)
    model_odd = fs.box_model(odd_time, 0, duration, odd_depth)
    model_even = fs.box_model(even_time, 0, duration, even_depth)

    fig, axes = plt.subplots(1, 2, figsize=(15,4))
    axes[0].plot(odd_time, odd_flux, '.')
    axes[0].plot(odd_time, model_odd, '-')
    axes[1].plot(even_time, even_flux, '.')
    axes[1].plot(even_time, model_even, '-')
    plt.show()

    print(odd_depth-even_depth)
    # time_small = time[:]
    # time, flux = preprocess.phase_fold_and_sort_light_curve(time, flux_big, period, t0)
    #
    # time_small, flux_small = preprocess.phase_fold_and_sort_light_curve(time_small, flux_small, period, t0)
    #
    # t0, new_time, new_flux, depth = fs.find_secondary(time, flux, duration, period)
    # model = fs.box_model(new_time, t0, duration, depth)
    # fig = plt.figure()
    # plt.plot(new_time, new_flux, '.')
    # plt.plot(new_time, model, 'r-')
    # plt.show()

    # primary_depth = preprocess.measure_eclipse_depth(time_small, flux_small, duration, Qingress)
    # primary_depth_big = preprocess.measure_eclipse_depth(time, flux, duration, Qingress)
    #
    # model1 = fs.box_model(time, 0, duration, primary_depth)
    # model2 = fs.box_model(time, 0, duration, primary_depth_big)
    #
    # OOT_flux = flux_small[np.where(abs(time_small) > duration / 2)]
    # std = np.std(OOT_flux)
    # print(primary_depth, primary_depth_big)
    # print("depth change=", (primary_depth_big - primary_depth) / std)
    #
    # fig = plt.figure()
    # plt.title('Primary transit')
    # plt.plot(time, flux, 'k.', alpha=0.3)
    # plt.plot(time, model1, 'r-')
    #
    # plt.plot(time_small, flux_small, 'b.', alpha=0.3)
    # plt.plot(time_small, model2, 'c--')
    #
    # plt.show()


if __name__ == '__main__':
    tic = int(sys.argv[1])
    tess_data_dir = '/Users/liangyu/Documents/astronet-triage/astronet/astronet/tess'
    input_tce_csv_file = 'astronet/tces.csv'
    test_secondary(tic, input_tce_csv_file)
